

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from pathlib import Path

ROOT_DIR  = Path(r"C:\Users\summer.intern12\Downloads\digital_twin_consumer_pipeline")
DOWNLOADS = Path.home() / "Downloads"

# FIX 1: Updated keys to perfectly match our active live production models on Groq
MODEL_SCALE_PROXY = {
    "llama3 (local, Ollama)":      8,     # 8B
    "llama-3.1-8b-instant":        8,     # 8B
    "llama-3.3-70b-versatile":     70,    # 70B
}


def calculate_jsd(p, q) -> float:
    p = np.array(p, dtype=np.float64); q = np.array(q, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        raise ValueError("A distribution sums to zero — cannot compute JSD.")
    p = p / p.sum(); q = q / q.sum()
    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            res = a * np.log2(a / b); res[~np.isfinite(res)] = 0.0
        return np.sum(res)
    m = 0.5 * (p + q)
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def find_file(candidates, label, required=True):
    for path in candidates:
        if path.exists():
            return path
    if required:
        print(f"[FATAL] Could not find {label}.")
        sys.exit(1)
    return None


def main():
    print("\n=== Model Ladder Comparison (verified data only) ===\n")

    human_path = find_file([
        DOWNLOADS / "real_telecom_survey.csv",
        ROOT_DIR / "real_telecom_survey.csv",
        ROOT_DIR / "data" / "raw" / "real_telecom_survey.csv"
    ], "human baseline file")
    human_df = pd.read_csv(human_path)
    all_options = sorted(human_df["human_choice"].dropna().unique())
    human_dist = human_df["human_choice"].value_counts(normalize=True)
    p_vals = [human_dist.get(opt, 0.0) for opt in all_options]
    print(f"Human baseline (n={len(human_df)}): "
          + ", ".join(f"{o}={p:.3f}" for o, p in zip(all_options, p_vals)))

    rows = []

    # ── local llama3 result ─────────────────────────────────────────────────
    local_path = find_file([ROOT_DIR / "outputs" / "calibrated_experiment_results.csv"],
                           "local llama3 results", required=False)
    if local_path:
        local_df = pd.read_csv(local_path)
        local_df = local_df[local_df["parse_success"] == True]
        if len(local_df) > 0:
            dist = local_df["choice"].value_counts(normalize=True)
            q_vals = [dist.get(opt, 0.0) for opt in all_options]
            jsd = calculate_jsd(p_vals, q_vals)
            expected = np.array(p_vals) * len(local_df)
            observed = np.array([local_df["choice"].value_counts().get(opt, 0) for opt in all_options])
            chi2, pval = chisquare(f_obs=observed, f_exp=expected) if (expected >= 5).all() else (None, None)
            rows.append({
                "model": "llama3 (local, Ollama)", "n": len(local_df),
                "jsd": jsd, "fidelity_pct": (1 - jsd) * 100,
                "p_value": pval, "scale_b_params": MODEL_SCALE_PROXY.get("llama3 (local, Ollama)"),
                **{f"dist_{o}": d for o, d in zip(all_options, q_vals)},
            })

    # ── Groq cross-model results ────────────────────────────────────────────
    groq_path = find_file([ROOT_DIR / "outputs" / "cross_model_comparison_results_FIXED.csv",
                           ROOT_DIR / "outputs" / "cross_model_comparison_results.csv"],
                          "Groq cross-model results", required=False)
    if groq_path:
        groq_df = pd.read_csv(groq_path)
        
        # FIX 2: Dynamic column detection to safely prevent Column KeyErrors
        model_col = "model_tested" if "model_tested" in groq_df.columns else "model"
        choice_col = "simulated_choice" if "simulated_choice" in groq_df.columns else "choice"
        
        for model_name in groq_df[model_col].unique():
            sub = groq_df[(groq_df[model_col] == model_name) & (groq_df["parse_success"] == True)]
            if len(sub) == 0:
                print(f"\n[SKIP] {model_name}: 0 successful trials, excluded from comparison.")
                continue
            dist = sub[choice_col].value_counts(normalize=True)
            q_vals = [dist.get(opt, 0.0) for opt in all_options]
            jsd = calculate_jsd(p_vals, q_vals)
            expected = np.array(p_vals) * len(sub)
            observed = np.array([sub[choice_col].value_counts().get(opt, 0) for opt in all_options])
            chi2, pval = chisquare(f_obs=observed, f_exp=expected) if (expected >= 5).all() else (None, None)
            rows.append({
                "model": model_name, "n": len(sub),
                "jsd": jsd, "fidelity_pct": (1 - jsd) * 100,
                "p_value": pval, "scale_b_params": MODEL_SCALE_PROXY.get(model_name, np.nan),
                **{f"dist_{o}": d for o, d in zip(all_options, q_vals)},
            })

    if not rows:
        print("\n[FATAL] No usable model results found. Run the experiment scripts first.")
        sys.exit(1)

    result_df = pd.DataFrame(rows).sort_values("scale_b_params")
    out_csv = ROOT_DIR / "outputs" / "model_ladder_final.csv"
    result_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("MODEL LADDER — VERIFIED RESULTS ONLY")
    print("=" * 70)
    for _, r in result_df.iterrows():
        sig = (f"p={r['p_value']:.4f}" if pd.notna(r["p_value"]) else "n/a (low expected counts)")
        warn = "  [SMALL SAMPLE]" if r["n"] < 30 else ""
        print(f"{r['model']:<28} n={r['n']:<4} JSD={r['jsd']:.4f}  "
              f"fidelity={r['fidelity_pct']:.2f}%  chi2 {sig}{warn}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Telecom Consumer Alignment — Model Ladder (verified, no fallback data)",
                 fontsize=14, fontweight="bold")

    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(result_df)))
    bars = ax1.bar(result_df["model"], result_df["fidelity_pct"], color=colors)
    ax1.set_ylabel("Fidelity (%) = (1 - JSD) \u00d7 100")
    ax1.set_title("Fidelity by model")
    ax1.set_ylim(0, 105)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    for bar, n in zip(bars, result_df["n"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"n={n}", ha="center", fontsize=9)

    ax2 = axes[1]
    valid_scale = result_df.dropna(subset=["scale_b_params"])
    ax2.scatter(valid_scale["scale_b_params"], valid_scale["fidelity_pct"],
               s=120, c=colors[:len(valid_scale)], zorder=5)
    for _, r in valid_scale.iterrows():
        ax2.annotate(r["model"], (r["scale_b_params"], r["fidelity_pct"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Model scale (B parameters, log scale)")
    ax2.set_ylabel("Fidelity (%)")
    ax2.set_title("Does fidelity scale with model size?\n(Llama family only)")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    out_plot = ROOT_DIR / "outputs" / "model_ladder_final_plot.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_plot}")
    print(f"CSV saved:  {out_csv}")


if __name__ == "__main__":
    main()