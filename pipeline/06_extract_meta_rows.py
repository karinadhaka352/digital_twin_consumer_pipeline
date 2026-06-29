import pandas as pd
import numpy as np
from pathlib import Path

def extract_clean_construct_correlations():
    print("\n=== Automated Meta-Row Extraction Engine ===")
    ROOT_DIR = Path(__file__).parent.parent
    data_file = ROOT_DIR / "data" / "meta_analysis" / "raw_data" / "AI Chatbot Affordances and Purchase Intention Dataset.xlsx"
    
    if not data_file.exists():
        print(f"[ERROR] Cannot find dataset at {data_file}")
        return

    # Ingest raw data
    df = pd.read_excel(data_file)
    n_subjects = len(df)
    
    # Calculate the comprehensive raw correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    
    # Group item lists by their construct prefix tags
    ca_cols = [c for c in df.columns if c.startswith("CA")]
    ce_cols = [c for c in df.columns if c.startswith("CE")]
    ct_cols = [c for c in df.columns if c.startswith("CT")]
    cs_cols = [c for c in df.columns if c.startswith("CS")]
    pi_cols = [c for c in df.columns if c.startswith("PI")]
    
    constructs = {
        "Chatbot Affordance": ca_cols,
        "Customer Engagement": ce_cols,
        "Customer Trust": ct_cols,
        "Customer Satisfaction": cs_cols,
        "Purchase Intention": pi_cols
    }
    
    print(f"Processing data for N = {n_subjects} participants across identified construct blocks...")
    print("\n" + "="*70)
    print(f"{'Construct 1':<22} | {'Construct 2':<22} | {'Pooled r':<10}")
    print("="*70)
    
    # Extract the average correlation between different construct families
    keys = list(constructs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c1_name, c1_list = keys[i], constructs[keys[i]]
            c2_name, c2_list = keys[j], constructs[keys[j]]
            
            if not c1_list or not c2_list:
                continue
                
            # Extract the subset correlation matrix between the two blocks
            sub_corr = corr_matrix.loc[c1_list, c2_list].values
            
            # Use Fisher's Z transformation to average the correlations safely
            z_values = 0.5 * np.log((1 + sub_corr) / (1 - sub_corr))
            avg_z = np.mean(z_values)
            avg_r = (np.exp(2 * avg_z) - 1) / (np.exp(2 * avg_z) + 1)
            
            print(f"{c1_name:<22} | {c2_name:<22} | {avg_r:.3f}")
            
if __name__ == "__main__":
    extract_clean_construct_correlations()