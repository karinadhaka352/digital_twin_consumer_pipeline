import pandas as pd
import numpy as np
from pathlib import Path

def extract_and_save_meta_rows():
    print("\n=== Automated Meta-Row Extraction & Storage Engine ===")
    ROOT_DIR = Path(__file__).parent.parent
    
    # Input path
    file_name = "AI Chatbot Affordances and Purchase Intention Dataset.xlsx"
    data_file = ROOT_DIR / "data" / "meta_analysis" / "raw_data" / file_name
    
    # Output path for permanent storage
    output_file = ROOT_DIR / "outputs" / "extracted_meta_study_records.csv"
    
    if not data_file.exists():
        print(f"[ERROR] Cannot find dataset at {data_file}")
        return

    # Ingest raw data
    df = pd.read_excel(data_file)
    n_subjects = len(df)
    corr_matrix = df.corr(numeric_only=True)
    
    # Define construct groups
    constructs = {
        "Chatbot Affordance": [c for c in df.columns if c.startswith("CA")],
        "Customer Engagement": [c for c in df.columns if c.startswith("CE")],
        "Customer Trust": [c for c in df.columns if c.startswith("CT")],
        "Customer Satisfaction": [c for c in df.columns if c.startswith("CS")],
        "Purchase Intention": [c for c in df.columns if c.startswith("PI")]
    }
    
    extracted_records = []
    keys = list(constructs.keys())
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c1_name, c1_list = keys[i], constructs[keys[i]]
            c2_name, c2_list = keys[j], constructs[keys[j]]
            
            if not c1_list or not c2_list:
                continue
                
            sub_corr = corr_matrix.loc[c1_list, c2_list].values
            
            # Fisher's Z transformation to pool correlation coefficients safely
            z_values = 0.5 * np.log((1 + sub_corr) / (1 - sub_corr))
            avg_z = np.mean(z_values)
            avg_r = (np.exp(2 * avg_z) - 1) / (np.exp(2 * avg_z) + 1)
            
            # Save record structure
            extracted_records.append({
                "Study_Source": file_name,
                "Construct_1": c1_name,
                "Construct_2": c2_name,
                "Correlation_r": round(avg_r, 4),
                "Sample_Size_n": n_subjects
            })
            
    # Convert list to DataFrame
    output_df = pd.DataFrame(extracted_records)
    
    # Save to your hard drive permanently
    output_df.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Extracted 10 macro-relationships from raw data.")
    print(f"[STORED PERMANENTLY AT]: {output_file}")
    
    # Print preview matrix on screen
    print("\nPreview of saved database rows:")
    print(output_df.head(4))

if __name__ == "__main__":
    extract_and_save_meta_rows()