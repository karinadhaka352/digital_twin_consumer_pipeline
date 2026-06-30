import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR    = Path(r"C:\Users\summer.intern12\Downloads\digital_twin_consumer_pipeline")
DOWNLOADS   = Path(r"C:\Users\summer.intern12\Downloads")


def calculate_jsd(p, q) -> float:
    """Jensen-Shannon Divergence between two discrete distributions."""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        raise ValueError(
            f"Cannot compute JSD: one distribution sums to zero. "
            f"p={p.tolist()}, q={q.tolist()}. This usually means the "
            f"'choice' or 'human_choice' column had no valid values."
        )
    p = p / p.sum()
    q = q / q.sum()

    def kl_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            res = a * np.log2(a / b)
            res[~np.isfinite(res)] = 0.0
        return np.sum(res)

    m = 0.5 * (p + q)
    jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return float(jsd)


def find_file(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            print(f"  Found {label} at: {path}")
            return path
    print(f"\n[FATAL] Could not find {label}.")
    print(f"  Looked in:")
    for path in candidates:
        print(f"    - {path}")
    print(f"  Fix: confirm the file exists at one of these paths, or edit")
    print(f"  the candidate list at the top of this script.")
    sys.exit(1)


def evaluate_telecom_alignment():
    print("\n" + "=" * 60)
    print("TELECOM ALIGNMENT EVALUATION — full diagnostics, no fallback data")
    print("=" * 60)

    # ── 1. Load human baseline (required, no substitute) ──────────────────
    human_path = find_file([
        DOWNLOADS / "real_telecom_survey.csv",
        ROOT_DIR / "real_telecom_survey.csv",
        ROOT_DIR / "data" / "raw" / "real_telecom_survey.csv",
    ], "human baseline file (real_telecom_survey.csv)")
    human_df = pd.read_csv(human_path)

    print(f"\n  Human baseline rows: {len(human_df)}")
    if "human_choice" not in human_df.columns:
        print(f"  [FATAL] 'human_choice' column not found. Columns present: {list(human_df.columns)}")
        sys.exit(1)
    print(f"  human_choice value counts:\n{human_df['human_choice'].value_counts().to_string()}")

    # ── 2. Load agent results (required, no substitute) ───────────────────
    agent_path = find_file([
        ROOT_DIR / "outputs" / "calibrated_experiment_results.csv",
        ROOT_DIR / "pipeline" / "outputs" / "calibrated_experiment_results.csv",
        DOWNLOADS / "calibrated_experiment_results.csv",
    ], "agent results file (calibrated_experiment_results.csv)")
    agent_df = pd.read_csv(agent_path)
    save_dir = agent_path.parent

    print(f"\n  Agent results rows (before filtering): {len(agent_df)}")
    if "parse_success" not in agent_df.columns:
        print(f"  [FATAL] 'parse_success' column not found. Columns present: {list(agent_df.columns)}")
        sys.exit(1)

    n_total   = len(agent_df)
    n_success = int(agent_df["parse_success"].sum())
    print(f"  parse_success == True: {n_success} / {n_total} ({n_success/n_total:.1%})")

    if n_success == 0:
        print("\n  [FATAL] Zero successful trials. Cannot compute any alignment metric.")
        print("  Check the 'error' column in calibrated_experiment_results.csv for why parsing failed.")
        if "error" in agent_df.columns:
            print("\n  Sample errors:")
            print(agent_df[agent_df["parse_success"] == False]["error"].head(5).to_string())
        sys.exit(1)

    if n_success < 10:
        print(f"\n  [WARNING] Only {n_success} successful trials. Any percentage below is")
        print(f"  based on a very small sample and should be reported with that caveat —")
        print(f"  do not present this as a precise, stable estimate.")

    agent_df_success = agent_df[agent_df["parse_success"] == True]
    if "choice" not in agent_df_success.columns:
        print(f"  [FATAL] 'choice' column not found. Columns present: {list(agent_df_success.columns)}")
        sys.exit(1)

    n_valid_choice = agent_df_success["choice"].notna().sum()
    print(f"  Rows with a non-null 'choice' value: {n_valid_choice} / {n_success}")
    print(f"  choice value counts:\n{agent_df_success['choice'].value_counts().to_string()}")

    # ── 3. Build distributions ──────────────────────────────────────────────
    human_dist = human_df["human_choice"].value_counts(normalize=True).sort_index()
    agent_dist = agent_df_success["choice"].value_counts(normalize=True).sort_index()

    all_options = sorted(set(human_df["human_choice"].dropna().unique()) |
                          set(agent_df_success["choice"].dropna().unique()))
    if len(all_options) == 0:
        print("\n  [FATAL] No valid choice options found in either dataset.")
        sys.exit(1)

    p_vals = [human_dist.get(opt, 0.0) for opt in all_options]
    q_vals = [agent_dist.get(opt, 0.0) for opt in all_options]

    print(f"\n  Distribution comparison:")
    print(f"  {'Option':<15} {'Human P':>10} {'Agent Q':>10}")
    for opt, p, q in zip(all_options, p_vals, q_vals):
        print(f"  {opt:<15} {p:>10.4f} {q:>10.4f}")

    # ── 4. Compute JSD + fidelity, with sanity check ────────────────────────
    jsd_score = calculate_jsd(p_vals, q_vals)
    fidelity_pct = (1.0 - jsd_score) * 100

    if not (0.0 <= fidelity_pct <= 100.0):
        print(f"\n  [FATAL] Computed fidelity ({fidelity_pct:.2f}%) is outside the valid 0-100% range.")
        print(f"  This means JSD itself is invalid (JSD={jsd_score}). JSD should always be")
        print(f"  between 0 and 1 (using log base 2). A value outside that range means")
        print(f"  something upstream is wrong — check for NaN values, mismatched option")
        print(f"  labels (e.g. 'Option A' vs 'option a'), or a sum-to-zero distribution.")
        sys.exit(1)

    n_with_sample_warning = " (SMALL SAMPLE — see warning above)" if n_success < 10 else ""

    print("\n" + "=" * 60)
    print("RESULTS — TELECOM ENVIRONMENT")
    print("=" * 60)
    print(f"Models Evaluated      : ['llama3']")
    print(f"Successful trials     : {n_success} / {n_total}{n_with_sample_warning}")
    print(f"Mathematical JSD       : {jsd_score:.4f}")
    print(f"Fidelity Rating        : {fidelity_pct:.2f}%")
    print("=" * 60)

    # ── 5. Uncalibrated baseline comparison (only if we actually have it) ──
    uncalibrated_path = ROOT_DIR / "outputs" / "baseline_results.csv"
    uncalibrated_jsd = None
    if uncalibrated_path.exists():
        try:
            base_df = pd.read_csv(uncalibrated_path)
            base_df = base_df[base_df.get("parse_success", True) == True]
            base_dist = base_df["choice"].value_counts(normalize=True).sort_index()
            base_q = [base_dist.get(opt, 0.0) for opt in all_options]
            uncalibrated_jsd = calculate_jsd(p_vals, base_q)
            print(f"\n  Uncalibrated baseline JSD (from your own data): {uncalibrated_jsd:.4f}")
        except Exception as e:
            print(f"\n  [WARNING] Found {uncalibrated_path} but could not score it: {e}")
    else:
        print(f"\n  [NOTE] No uncalibrated baseline file found at {uncalibrated_path}.")
        print(f"  The plot below will mark this bar as 'not measured' rather than reusing")
        print(f"  the base paper's number, which was not generated by this run.")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Telecom Consumer Alignment — llama3 (real run, no fallback data)",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(all_options))
    width = 0.35
    ax1.bar(x - width/2, p_vals, width, label=f"Human Baseline (n={len(human_df)})", color="#5B9BD5")
    ax1.bar(x + width/2, q_vals, width, label=f"Synthetic Twins (n={n_success})", color="#70AD47")
    ax1.set_ylabel("Probability Distribution Density")
    ax1.set_title("Choice Distribution Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_options)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    if uncalibrated_jsd is not None:
        bar_labels = ["Uncalibrated\n(your data)", "Calibrated\nTri-Agent"]
        bar_vals = [uncalibrated_jsd, jsd_score]
        bar_colors = ["#E46C0A", "#4F81BD"]
    else:
        bar_labels = ["Uncalibrated\n(not measured)", "Calibrated\nTri-Agent"]
        bar_vals = [0, jsd_score]
        bar_colors = ["#CCCCCC", "#4F81BD"]
    ax2.bar(bar_labels, bar_vals, color=bar_colors, width=0.5)
    if uncalibrated_jsd is None:
        ax2.text(0, 0.002, "N/A", ha="center", fontsize=11, color="#888888")
    ax2.set_ylabel("Jensen-Shannon Divergence (lower = better)")
    ax2.set_title("Divergence Comparison")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = save_dir / "telecom_calibration_plot_FIXED.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\n  Plot saved: {plot_path}")

    # ── 7. Text report — every number traceable, nothing invented ──────────
    report_path = save_dir / "telecom_hypothesis_summary_FIXED.txt"
    with open(report_path, "w") as f:
        f.write("=== TELECOM ALIGNMENT REPORT (verified, no fallback data) ===\n\n")
        f.write(f"Human baseline file : {human_path}\n")
        f.write(f"Human baseline rows : {len(human_df)}\n")
        f.write(f"Agent results file  : {agent_path}\n")
        f.write(f"Agent total trials  : {n_total}\n")
        f.write(f"Agent successful    : {n_success} ({n_success/n_total:.1%})\n")
        if n_success < 10:
            f.write(f"SAMPLE SIZE WARNING : only {n_success} successful trials \u2014 treat as preliminary\n")
        f.write(f"\nDistribution comparison:\n")
        f.write(f"{'Option':<15}{'Human P':>12}{'Agent Q':>12}\n")
        for opt, p, q in zip(all_options, p_vals, q_vals):
            f.write(f"{opt:<15}{p:>12.4f}{q:>12.4f}\n")
        f.write(f"\nJensen-Shannon Divergence : {jsd_score:.4f}\n")
        f.write(f"Fidelity Rating           : {fidelity_pct:.2f}%\n")
        if uncalibrated_jsd is not None:
            f.write(f"Uncalibrated JSD (own data): {uncalibrated_jsd:.4f}\n")
        else:
            f.write(f"Uncalibrated JSD          : not measured (no baseline_results.csv found)\n")
        f.write(f"\nEvery number above is computed directly from the files listed.\n")
        f.write(f"No hardcoded or fallback values were used in this report.\n")

    print(f"  Report saved: {report_path}")
    print("\nDone. Every number above came from real data — nothing was substituted.")


if __name__ == "__main__":
    evaluate_telecom_alignment()