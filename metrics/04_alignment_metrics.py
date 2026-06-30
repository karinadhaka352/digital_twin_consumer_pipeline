

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import entropy, ks_2samp, f_oneway
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Human baseline from ABxLab ICLR 2026 paper
# Humans show ~4-9pp sensitivity to nudges
HUMAN_NUDGE_BASELINE = {
    "none":         0.50,   # 50/50 baseline
    "authority":    0.54,   # +4pp
    "social_proof": 0.54,   # +4pp
    "best_seller":  0.55,   # +5pp
    "financial":    0.59,   # +9pp
}

# Model capability proxy (Chatbot Arena ELO approximation)
MODEL_ELO = {
    "phi3":    1000,
    "llama3":  1150,
    "mistral": 1200,
    # Add more if you pull them:
    # "llama3.1:70b": 1280,
    # "gemma2":       1220,
}


# ═════════════════════════════════════════════════════════════
# H1 — KL-Divergence (Distributional Alignment)
# ═════════════════════════════════════════════════════════════
def compute_kl_divergence(df: pd.DataFrame, real_personas: pd.DataFrame) -> pd.DataFrame:
    """
    Compares distribution of synthetic consumer choices to
    expected distribution from real persona data.
    Lower KL = better alignment.
    """
    results = []
    models = df["model"].unique()

    # Real distribution: expected choice-A rate based on price sensitivity
    # High price sensitivity → should choose cheaper product B
    real_price_sens = real_personas["price_elasticity"].values

    for model in models:
        sub = df[(df["model"] == model) & (df["parse_success"] == True)]
        if len(sub) < 10:
            continue

        for nudge in df["nudge"].unique():
            ns = sub[sub["nudge"] == nudge]
            if len(ns) < 5:
                continue

            # Synthetic distribution: observed choice A rate per persona bucket
            choice_a_rate = (ns["choice"] == "A").mean()

            # Real expected: consumers with high price elasticity should prefer B (cheaper)
            # So expected A rate ~ 1 - mean_price_elasticity of these personas
            expected_a_rate = 1 - ns["price_elasticity"].mean()

            # KL divergence between two Bernoulli distributions
            p = np.clip([choice_a_rate, 1 - choice_a_rate], 1e-9, 1)
            q = np.clip([expected_a_rate, 1 - expected_a_rate], 1e-9, 1)
            kl = float(entropy(p, q))

            results.append({
                "model": model,
                "nudge": nudge,
                "metric": "KL_divergence",
                "value": round(kl, 4),
                "interpretation": "lower=better",
                "hypothesis": "H1"
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════
# H2 — LBE Index (Behavioral Realism)
# ═════════════════════════════════════════════════════════════
def compute_lbe_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Local Behavioral Equivalence Index.
    Measures whether synthetic consumers with similar persona profiles
    make similar choices (internal consistency = behavioral realism).
    Range: 0-1, higher = more realistic.
    """
    results = []
    models = df["model"].unique()

    for model in models:
        sub = df[(df["model"] == model) & (df["parse_success"] == True)].copy()
        if len(sub) < 10:
            continue

        # Bin consumers by price elasticity quartile
        sub["pe_quartile"] = pd.qcut(sub["price_elasticity"], q=4, labels=["Q1","Q2","Q3","Q4"],
                                      duplicates="drop")

        # Within each quartile, expect consistent choice direction
        consistency_scores = []
        for q in sub["pe_quartile"].unique():
            group = sub[sub["pe_quartile"] == q]
            if len(group) < 3:
                continue
            # High price sensitivity (Q3/Q4) should prefer B (cheaper)
            choice_a = (group["choice"] == "A").mean()
            is_high_pe = q in ["Q3","Q4"]
            # Expected: high PE → low A rate (prefer cheaper B)
            expected = 0.3 if is_high_pe else 0.7
            # Consistency = 1 - deviation from expectation
            consistency = 1 - abs(choice_a - expected)
            consistency_scores.append(consistency)

        lbe = np.mean(consistency_scores) if consistency_scores else 0.5

        # Also factor in confidence consistency
        conf_std = sub["confidence"].std() / 10 if "confidence" in sub.columns else 0.3
        lbe_adjusted = lbe * (1 - conf_std * 0.2)   # penalize high variance

        results.append({
            "model":  model,
            "nudge":  "all",
            "metric": "LBE_index",
            "value":  round(float(lbe_adjusted), 4),
            "interpretation": "higher=better",
            "hypothesis": "H2"
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════
# H3 — Nudge Gap Δ (Susceptibility vs Human Baseline)
# ═════════════════════════════════════════════════════════════
def compute_nudge_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures how much nudges shift synthetic consumer choices
    versus the human baseline from ABxLab paper.
    Lower gap = more human-like nudge calibration.
    """
    results = []
    models = df["model"].unique()

    for model in models:
        sub = df[(df["model"] == model) & (df["parse_success"] == True)]
        if len(sub) < 10:
            continue

        baseline_no_nudge = (sub[sub["nudge"] == "none"]["choice"] == "A").mean()

        for nudge, human_rate in HUMAN_NUDGE_BASELINE.items():
            if nudge == "none":
                continue
            ns = sub[sub["nudge"] == nudge]
            if len(ns) < 3:
                continue

            synthetic_rate = (ns["choice"] == "A").mean()
            synthetic_shift = synthetic_rate - baseline_no_nudge

            human_baseline_no_nudge = HUMAN_NUDGE_BASELINE["none"]
            human_shift = human_rate - human_baseline_no_nudge

            # Nudge gap = |synthetic shift - human shift| in percentage points
            gap_pp = abs(synthetic_shift - human_shift) * 100

            results.append({
                "model":          model,
                "nudge":          nudge,
                "metric":         "nudge_gap_pp",
                "value":          round(gap_pp, 2),
                "synthetic_shift_pp": round(synthetic_shift * 100, 2),
                "human_shift_pp":     round(human_shift * 100, 2),
                "interpretation": "lower=more human-like",
                "hypothesis":     "H3"
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════
# H4 — RAG Lift
# ═════════════════════════════════════════════════════════════
def compute_rag_lift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares alignment (LBE proxy) with vs without RAG grounding.
    Positive lift = RAG improves persona fidelity.
    """
    results = []
    models = df["model"].unique()

    for model in models:
        for rag in ["with_rag", "no_rag"]:
            sub = df[(df["model"] == model) &
                     (df["rag_condition"] == rag) &
                     (df["parse_success"] == True)]
            if len(sub) < 5:
                continue

            # Proxy: does choice align with price elasticity?
            # High PE consumer choosing B (cheaper) = aligned
            aligned = 0
            for _, row in sub.iterrows():
                is_high_pe = row.get("price_elasticity", 0.5) > 0.5
                chose_b    = row.get("choice") == "B"
                if is_high_pe == chose_b:
                    aligned += 1
            alignment = aligned / len(sub)

            results.append({
                "model":  model,
                "nudge":  "all",
                "rag":    rag,
                "metric": "rag_alignment",
                "value":  round(alignment, 4),
                "interpretation": "higher=better",
                "hypothesis": "H4"
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════
# Composite Alignment Score → 90% Threshold
# ═════════════════════════════════════════════════════════════
def compute_composite_alignment(kl_df, lbe_df, nudge_df, rag_df) -> pd.DataFrame:
    """
    Composite = 0.40*(1-norm_KL) + 0.30*LBE + 0.20*(1-norm_nudge_gap) + 0.10*RAG_lift
    Target: ≥ 0.90 = Digital Twin fidelity threshold
    """
    models = kl_df["model"].unique() if len(kl_df) > 0 else []
    results = []

    # Normalize KL (0-1 where 0=perfect)
    if len(kl_df) > 0:
        max_kl = kl_df["value"].max() or 1
        kl_by_model = kl_df.groupby("model")["value"].mean()
    else:
        max_kl = 1
        kl_by_model = pd.Series()

    if len(nudge_df) > 0:
        max_gap = nudge_df["value"].max() or 1
        gap_by_model = nudge_df.groupby("model")["value"].mean()
    else:
        max_gap = 1
        gap_by_model = pd.Series()

    lbe_by_model = lbe_df.groupby("model")["value"].mean() if len(lbe_df) > 0 else pd.Series()
    rag_by_model = (rag_df[rag_df["rag"]=="with_rag"].groupby("model")["value"].mean()
                    if len(rag_df) > 0 else pd.Series())

    for model in set(list(kl_by_model.index) + list(lbe_by_model.index)):
        kl_norm  = 1 - (kl_by_model.get(model, 0.3) / max_kl)
        lbe      = lbe_by_model.get(model, 0.5)
        gap_norm = 1 - (gap_by_model.get(model, 50) / max(max_gap, 1))
        rag      = rag_by_model.get(model, 0.5)

        composite = (0.40 * kl_norm +
                     0.30 * lbe     +
                     0.20 * gap_norm+
                     0.10 * rag)

        results.append({
            "model":           model,
            "kl_component":    round(kl_norm, 3),
            "lbe_component":   round(lbe, 3),
            "nudge_component": round(gap_norm, 3),
            "rag_component":   round(rag, 3),
            "composite_score": round(composite, 3),
            "meets_90pct":     composite >= 0.90,
            "elo_proxy":       MODEL_ELO.get(model, 1100),
        })

    return pd.DataFrame(results).sort_values("composite_score", ascending=False)


# ═════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════
def plot_model_ladder(composite_df: pd.DataFrame):
    """Main figure: Model quality → alignment score (your paper's key plot)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Digital Twin Consumer: Model Quality vs Alignment Score",
                 fontsize=14, fontweight="bold")

    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

    # Left: composite score by model
    ax = axes[0]
    bars = ax.barh(composite_df["model"], composite_df["composite_score"],
                   color=colors[:len(composite_df)], alpha=0.85)
    ax.axvline(x=0.90, color="red", linestyle="--", linewidth=2, label="90% threshold")
    ax.set_xlabel("Composite Alignment Score", fontsize=11)
    ax.set_title("Overall Digital Twin Fidelity", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.legend()
    for bar, score in zip(bars, composite_df["composite_score"]):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=9)

    # Right: model capability ELO vs composite (the ladder plot)
    ax2 = axes[1]
    elos   = composite_df["elo_proxy"].values
    scores = composite_df["composite_score"].values
    ax2.scatter(elos, scores, s=150, c=colors[:len(composite_df)], zorder=5)
    for i, row in composite_df.iterrows():
        ax2.annotate(row["model"],
                     (row["elo_proxy"], row["composite_score"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
    if len(elos) > 1:
        z = np.polyfit(elos, scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(elos)-50, max(elos)+50, 100)
        ax2.plot(x_line, p(x_line), "k--", alpha=0.4, label="trend")
    ax2.axhline(y=0.90, color="red", linestyle="--", linewidth=2, label="90% threshold")
    ax2.set_xlabel("Model Capability (ELO proxy)", fontsize=11)
    ax2.set_ylabel("Composite Alignment Score", fontsize=11)
    ax2.set_title("Model Quality Ladder\n(H1: Better models → better Digital Twins)", fontsize=12)
    ax2.legend()

    plt.tight_layout()
    out = OUT_DIR / "model_ladder_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]✓ Saved: {out}[/]")


def plot_nudge_gap(nudge_df: pd.DataFrame):
    """Nudge susceptibility gap vs human baseline."""
    if len(nudge_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = nudge_df.pivot_table(index="nudge", columns="model", values="value")
    x     = np.arange(len(pivot))
    width = 0.8 / max(len(pivot.columns), 1)

    for i, model in enumerate(pivot.columns):
        ax.bar(x + i*width, pivot[model], width, label=model, alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(pivot.columns)-1) / 2)
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel("Nudge Gap vs Human Baseline (pp)")
    ax.set_title("H3: Nudge Susceptibility Gap\n(Lower = more human-like | Human baseline: ~4-9pp)")
    ax.legend()
    ax.axhspan(0, 9, alpha=0.1, color="green", label="Human range")

    out = OUT_DIR / "nudge_gap_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]✓ Saved: {out}[/]")


def plot_component_breakdown(composite_df: pd.DataFrame):
    """Stacked bar of composite score components."""
    if len(composite_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    components = ["kl_component","lbe_component","nudge_component","rag_component"]
    weights    = [0.40, 0.30, 0.20, 0.10]
    labels     = ["KL-div (40%)","LBE index (30%)","Nudge gap (20%)","RAG lift (10%)"]
    colors_c   = ["#4C72B0","#55A868","#DD8452","#8172B2"]

    bottom = np.zeros(len(composite_df))
    for comp, w, label, color in zip(components, weights, labels, colors_c):
        vals = composite_df[comp].values * w
        ax.bar(composite_df["model"], vals, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += vals

    ax.axhline(y=0.90, color="red", linestyle="--", lw=2, label="90% threshold")
    ax.set_ylabel("Composite Alignment Score")
    ax.set_title("Alignment Score Components by Model\n(Digital Twin Fidelity Breakdown)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1.05)

    out = OUT_DIR / "component_breakdown_plot.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"  [green]✓ Saved: {out}[/]")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results_path = OUT_DIR / "experiment_results.csv"
    if not results_path.exists():
        console.print("[red]Run Step 3 first: python pipeline/03_run_experiments.py[/]")
        exit(1)

    console.print("\n[bold cyan]═══ Computing Alignment Metrics ═══[/]\n")

    df      = pd.read_csv(results_path)
    personas = pd.read_csv("data/persona_vectors.csv")
    success = df[df["parse_success"] == True].copy()

    console.print(f"  Loaded {len(df)} trials, {len(success)} successful ({len(success)/len(df):.1%})")
    console.print(f"  Models: {success['model'].unique().tolist()}")

    # Compute all metrics
    console.print("\n[bold]Computing H1: KL-Divergence...[/]")
    kl_df = compute_kl_divergence(success, personas)

    console.print("[bold]Computing H2: LBE Index...[/]")
    lbe_df = compute_lbe_index(success)

    console.print("[bold]Computing H3: Nudge Gap...[/]")
    nudge_df = compute_nudge_gap(success)

    console.print("[bold]Computing H4: RAG Lift...[/]")
    rag_df = compute_rag_lift(df)

    console.print("[bold]Computing Composite Alignment Score...[/]")
    composite_df = compute_composite_alignment(kl_df, lbe_df, nudge_df, rag_df)

    # Save all metrics
    all_metrics = pd.concat([kl_df, lbe_df, nudge_df, rag_df], ignore_index=True)
    all_metrics.to_csv(OUT_DIR / "alignment_scores.csv", index=False)
    composite_df.to_csv(OUT_DIR / "composite_alignment.csv", index=False)

    # Plots
    console.print("\n[bold]Generating plots...[/]")
    plot_model_ladder(composite_df)
    plot_nudge_gap(nudge_df)
    plot_component_breakdown(composite_df)

    # Print hypothesis summary
    console.print("\n[bold cyan]═══ HYPOTHESIS RESULTS ═══[/]\n")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Hypothesis", style="bold")
    table.add_column("Metric")
    table.add_column("Best Model")
    table.add_column("Result")
    table.add_column("Supported?")

    if len(kl_df) > 0:
        best_h1 = kl_df.groupby("model")["value"].mean().idxmin()
        kl_val  = kl_df.groupby("model")["value"].mean().min()
        table.add_row("H1: Distrib. Alignment", "KL-divergence↓",
                      best_h1, f"{kl_val:.4f}", "✓" if kl_val < 0.3 else "~")

    if len(lbe_df) > 0:
        best_h2 = lbe_df["model"].iloc[lbe_df["value"].argmax()]
        lbe_val = lbe_df["value"].max()
        table.add_row("H2: Behavioral Realism", "LBE index↑",
                      best_h2, f"{lbe_val:.3f}", "✓" if lbe_val > 0.7 else "~")

    if len(nudge_df) > 0:
        best_h3 = nudge_df.groupby("model")["value"].mean().idxmin()
        gap_val = nudge_df.groupby("model")["value"].mean().min()
        table.add_row("H3: Nudge Calibration", "Nudge gap↓ (pp)",
                      best_h3, f"{gap_val:.1f}pp", "✓" if gap_val < 20 else "~")

    if len(rag_df) > 0:
        rag_wide  = rag_df[rag_df["rag"]=="with_rag"]["value"].mean()
        rag_none  = rag_df[rag_df["rag"]=="no_rag"]["value"].mean()
        lift      = rag_wide - rag_none
        table.add_row("H4: RAG Lift", "Alignment lift",
                      "all models", f"+{lift:.3f}", "✓" if lift > 0 else "✗")

    if len(composite_df) > 0:
        best_comp = composite_df.iloc[0]
        table.add_row("H5: 90% Threshold", "Composite score",
                      best_comp["model"],
                      f"{best_comp['composite_score']:.3f}",
                      "✓" if best_comp["meets_90pct"] else f"Closest: {best_comp['composite_score']:.3f}")

    console.print(table)

    # Save summary text
    summary = f"""HYPOTHESIS RESULTS SUMMARY
Generated from: {results_path}
Models tested: {success['model'].unique().tolist()}
Total trials: {len(df)} | Success rate: {len(success)/len(df):.1%}

Composite Alignment Scores:
{composite_df[['model','composite_score','meets_90pct','elo_proxy']].to_string(index=False)}

Key finding: {'Best model achieves ≥90% threshold' if composite_df['meets_90pct'].any() else 'No model achieves 90% yet — consider pulling larger models (llama3.1:70b)'}
"""
    with open(OUT_DIR / "hypothesis_summary.txt", "w") as f:
        f.write(summary)

    console.print(f"\n[bold green]✓ All metrics computed![/]")
    console.print(f"  Alignment scores → {OUT_DIR}/alignment_scores.csv")
    console.print(f"  Model ladder plot → {OUT_DIR}/model_ladder_plot.png")
    console.print(f"  Summary → {OUT_DIR}/hypothesis_summary.txt")
    console.print(f"\n[bold]Next:[/] jupyter notebook notebooks/05_results_analysis.ipynb")
