import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chisquare

def main():
    ROOT_DIR = Path(__file__).parent.parent
    
    agent_path = ROOT_DIR / "outputs" / "calibrated_experiment_results.csv"
    human_path = ROOT_DIR / "data" / "raw" / "real_telecom_survey.csv"
    
    if not agent_path.exists() or not human_path.exists():
        print("[ERROR] Missing input datasets. Make sure your experiment outputs exist.")
        return

    agent_df = pd.read_csv(agent_path)
    agent_df = agent_df[agent_df["parse_success"] == True]
    
    human_df = pd.read_csv(human_path)
    
    n_human = len(human_df)
    n_agent = len(agent_df)
    
    print("\n==================================================")
    print("           PER-OPTION SKEW SIGNIFICANCE TEST       ")
    print("==================================================")
    print(f"\nHuman baseline n = {n_human}")
    print(f"Agent sample n   = {n_agent}")
    
    all_options = sorted(list(set(human_df["human_choice"].dropna().unique())))
    
    human_props = human_df["human_choice"].value_counts(normalize=True)
    human_p = np.array([human_props.get(opt, 0.0) for opt in all_options])
    
    agent_counts = agent_df["choice"].value_counts()
    agent_observed = np.array([agent_counts.get(opt, 0) for opt in all_options], dtype=int)
    
    expected = human_p * n_agent
    
    print(f"\n{'Option':<12}{'Human P':>10}{'Expected':>10}{'Observed':>10}{'Rel. diff':>12}")
    for opt, p, exp, obs in zip(all_options, human_p, expected, agent_observed):
        rel_diff = ((obs - exp) / exp * 100) if exp > 0 else float("nan")
        print(f"{opt:<12}{p:>10.4f}{exp:>10.1f}{obs:>10d}{rel_diff:>11.1f}%")
        
    chi2, p_value = chisquare(f_obs=agent_observed, f_exp=expected)
    
    print(f"\n Chi-square statistic : {chi2:.3f}")
    print(f" Degrees of freedom   : {len(all_options) - 1}")
    
    if p_value < 0.0001:
        print(f" p-value              : < 0.0001")
    else:
        print(f" p-value              : {p_value:.4f}")
        
    alpha = 0.05
    if p_value < alpha:
        p_display = "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
        verdict = (f"REJECT H0 (p={p_display} < {alpha}): the agent's choice distribution "
                   f"differs significantly from the human baseline. The per-option skew is "
                   f"unlikely to be sampling noise.")
    else:
        p_display = f"{p_value:.4f}"
        verdict = (f"FAIL TO REJECT H0 (p={p_display} >= {alpha}): cannot rule out that the "
                   f"observed skew is sampling noise at this sample size.")
                   
    print(f"\n VERDICT: {verdict}")
    
    # Clean, structurally aligned file output writing logic
    out_path = agent_path.parent / "telecom_skew_significance.txt"
    with open(out_path, "w") as f:
        f.write("PER-OPTION SKEW SIGNIFICANCE TEST\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Human baseline file : {human_path} (n={n_human})\n")
        f.write(f"Agent results file  : {agent_path} (n={n_agent})\n\n")
        f.write(f"{'Option':<12}{'Human P':>10}{'Expected':>10}{'Observed':>10}{'Rel. diff':>12}\n")
        for opt, p, exp, obs in zip(all_options, human_p, expected, agent_observed):
            rel_diff = ((obs - exp) / exp * 100) if exp > 0 else float("nan")
            f.write(f"{opt:<12}{p:>10.4f}{exp:>10.1f}{obs:>10d}{rel_diff:>11.1f}%\n")
        f.write(f"\nChi-square statistic : {chi2:.3f}\n")
        f.write(f"Degrees of freedom   : {len(all_options) - 1}\n")
        p_txt = "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
        f.write(f"p-value              : {p_txt}\n")
        f.write(f"\nVERDICT: {verdict}\n")
        
    print(f"\n✓ Report successfully saved to: {out_path}")

if __name__ == "__main__":
    main()