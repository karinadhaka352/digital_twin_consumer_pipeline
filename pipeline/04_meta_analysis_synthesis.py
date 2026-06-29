import os
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_meta_analysis_pooling():
    print("\n=== Stages 4 to 10: Full Meta-Analysis Automation Suite ===")
    print("=" * 95)
    
    ROOT_DIR = Path(__file__).parent.parent
    data_file = ROOT_DIR / "data" / "meta_analysis" / "raw_data" / "Meta analysis sample.xlsx"
    outputs_dir = ROOT_DIR / "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    if not data_file.exists():
        print(f"[FATAL] Cannot locate spreadsheet data at: {data_file}")
        return

    df = pd.read_excel(data_file, sheet_name="Input")
    df_clean = df[(df["Include study"] == "Yes") & (df["Sufficient data"] == "Yes")].copy()
    
    # Mathematical transformations
    df_clean["fisher_z"] = 0.5 * np.log((1 + df_clean["Correlation"]) / (1 - df_clean["Correlation"]))
    df_clean["variance_z"] = 1 / (df_clean["Number of subjects"] - 3)
    df_clean["fe_weight"] = 1 / df_clean["variance_z"]
    df_clean["se_z"] = np.sqrt(df_clean["variance_z"])
    
    summary_names = []
    summary_fe_r = []
    summary_re_r = []
    
    for group_name, sub_df in df_clean.groupby("Subgroup"):
        k = len(sub_df)
        if k < 3:
            continue  
            
        # --- STAGE 5 & 6: POOLING & HETEROGENEITY ---
        sum_w = sub_df["fe_weight"].sum()
        weighted_z_sum = (sub_df["fisher_z"] * sub_df["fe_weight"]).sum()
        fe_pooled_z = weighted_z_sum / sum_w
        
        q_stat = (sub_df["fe_weight"] * ((sub_df["fisher_z"] - fe_pooled_z) ** 2)).sum()
        df_q = k - 1
        q_p_value = 1 - stats.chi2.cdf(q_stat, df_q)
        
        c_constant = sum_w - (sub_df["fe_weight"] ** 2).sum() / sum_w
        tau_squared = max(0.0, (q_stat - df_q) / c_constant)
        i_squared = max(0.0, (q_stat - df_q) / q_stat) * 100
        
        sub_df["re_weight"] = 1 / (sub_df["variance_z"] + tau_squared)
        baseline_re_pooled_z = (sub_df["fisher_z"] * sub_df["re_weight"]).sum() / sub_df["re_weight"].sum()
        baseline_re_pooled_r = (np.exp(2 * baseline_re_pooled_z) - 1) / (np.exp(2 * baseline_re_pooled_z) + 1)
        
        fe_pooled_r = (np.exp(2 * fe_pooled_z) - 1) / (np.exp(2 * fe_pooled_z) + 1)
        
        summary_names.append(group_name)
        summary_fe_r.append(fe_pooled_r)
        summary_re_r.append(baseline_re_pooled_r)
        
        # --- STAGE 8: PUBLICATION BIAS ---
        sub_df["std_effect"] = sub_df["fisher_z"] / sub_df["se_z"]
        sub_df["precision"] = 1 / sub_df["se_z"]
        slope, intercept = np.polyfit(sub_df["precision"], sub_df["std_effect"], 1)
        egger_p = 2 * (1 - stats.t.cdf(abs(intercept / (np.sqrt(np.sum((sub_df["std_effect"] - (slope * sub_df["precision"] + intercept))**2) / (k - 2)) * np.sqrt(1/k + (sub_df["precision"].mean()**2) / np.sum((sub_df["precision"] - sub_df["precision"].mean())**2)))), df=k-2))
        bias_status = "⚠️ BIAS DETECTED" if egger_p < 0.05 else "✅ CLEAN"

        print(f"\n📂 Subgroup Construct: {group_name}")
        print(f"  ├─ Studies (k) = {k:<4} | Random-Effects Pooled r = {baseline_re_pooled_r:.3f}")
        print(f"  ├─ Heterogeneity: I² = {i_squared:.1f}% | Bias Status: {bias_status}")
        
        # --- STAGE 9: AUTOMATED SENSITIVITY (LEAVE-ONE-OUT) ---
        max_deviation = 0.0
        outlier_study = None
        
        for idx in sub_df.index:
            # Leave one out loop
            loo_df = sub_df.drop(idx)
            
            # Recalculate Random Effects Pooled Estimate
            loo_sum_fe_w = loo_df["fe_weight"].sum()
            loo_fe_pooled_z = (loo_df["fisher_z"] * loo_df["fe_weight"]).sum() / loo_sum_fe_w
            loo_q = (loo_df["fe_weight"] * ((loo_df["fisher_z"] - loo_fe_pooled_z) ** 2)).sum()
            loo_c = loo_sum_fe_w - (loo_df["fe_weight"] ** 2).sum() / loo_sum_fe_w
            loo_tau2 = max(0.0, (loo_q - (len(loo_df) - 1)) / loo_c)
            
            loo_df["loo_re_weight"] = 1 / (loo_df["variance_z"] + loo_tau2)
            loo_re_z = (loo_df["fisher_z"] * loo_df["loo_re_weight"]).sum() / loo_df["loo_re_weight"].sum()
            loo_re_r = (np.exp(2 * loo_re_z) - 1) / (np.exp(2 * loo_re_z) + 1)
            
            # Calculate absolute drift from our baseline score
            deviation = abs(baseline_re_pooled_r - loo_re_r)
            if deviation > max_deviation:
                max_deviation = deviation
                outlier_study = df.loc[idx, "Study name"] if "Study name" in df.columns else f"Row {idx}"
                
        # Report stability analysis verdict
        stability = "🔴 UNSTABLE (High Leverage Study)" if max_deviation > 0.1 else "🟢 ROBUST"
        print(f"  └─ STAGE 9 SENSITIVITY: Max Delta = {max_deviation:.4f} via [{outlier_study}] -> {stability}")
        print("-" * 95)
        
    # --- STAGE 10: VISUALIZATION ---
    if len(summary_names) > 0:
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(summary_names))
        plt.scatter(summary_fe_r, y_pos - 0.15, color="gray", label="Fixed-Effects Pooled r", marker="o", s=80, alpha=0.6)
        plt.scatter(summary_re_r, y_pos + 0.15, color="crimson", label="Random-Effects Pooled r", marker="D", s=90)
        plt.yticks(y_pos, [name[:45] + "..." if len(name) > 45 else name for name in summary_names], fontsize=10)
        plt.axvline(x=0.0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Pooled Correlation Coefficient (r)", fontsize=12, fontweight="bold")
        plt.title("Master Meta-Analysis Subgroup Summary Chart\n(With Calibrated Analytical Models)", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        plt.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        plt.savefig(outputs_dir / "meta_analysis_summary_forest_plot.png", dpi=150)

if __name__ == "__main__":
    run_meta_analysis_pooling()