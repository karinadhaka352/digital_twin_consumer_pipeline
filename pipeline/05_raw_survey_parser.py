import pandas as pd
from pathlib import Path

def parse_raw_survey():
    print("\n=== Extracting Statistical Metrics from Raw Survey Data ===")
    ROOT_DIR = Path(__file__).parent.parent
    
    # Target a raw survey file (Example: AI Chatbot Affordances)
    file_name = "AI Chatbot Affordances and Purchase Intention Dataset.xlsx"
    data_file = ROOT_DIR / "data" / "meta_analysis" / "raw_data" / file_name
    
    if not data_file.exists():
        print(f"File not found: {data_file}")
        return

    # Ingest the survey data
    df = pd.read_excel(data_file)
    sample_size = len(df)
    
    print(f"Loaded '{file_name}' safely.")
    print(f"Total surveyed human subjects (n) = {sample_size}")
    
    # Calculate a clean Pearson correlation matrix across your numeric survey columns
    print("\nCalculating correlation values (r) for your survey variables...")
    corr_matrix = df.corr(numeric_only=True)
    
    # Display the correlations to use in your meta-analysis report
    print("\n--- Key Correlation Matrix (r Baseline) ---")
    print(corr_matrix.round(3))

if __name__ == "__main__":
    parse_raw_survey()