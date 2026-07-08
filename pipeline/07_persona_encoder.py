import pandas as pd
import json
from pathlib import Path

def encode_meta_to_persona():
    print("\n=== Initializing Behavioral Persona Encoder Module ===")
    ROOT_DIR = Path(__file__).parent.parent
    
    # Ingest the metrics you permanently saved earlier
    input_csv = ROOT_DIR / "outputs" / "extracted_meta_study_records.csv"
    output_json = ROOT_DIR / "outputs" / "encoded_agent_persona_rules.json"
    
    if not input_csv.exists():
        print(f"[ERROR] Run module 06 first. Missing baseline records at: {input_csv}")
        return

    # Read data nodes
    df = pd.read_csv(input_csv)
    
    # Filter for target relationship: Chatbot Affordance -> Purchase Intention
    target_rule = df[
        (df["Construct_1"] == "Chatbot Affordance") & 
        (df["Construct_2"] == "Purchase Intention")
    ]
    
    if target_rule.empty:
        print("[ERROR] Target behavioral pathway not found in records.")
        return
        
    correlation_r = float(target_rule["Correlation_r"].values[0])
    sample_size = int(target_rule["Sample_Size_n"].values[0])
    
    # Map raw statistical data into system constraint weights
    persona_config = {
        "persona_metadata": {
            "profile_type": "AI-Calibrated Synthetic Consumer",
            "empirical_basis": str(target_rule["Study_Source"].values[0]),
            "confidence_weight_n": sample_size
        },
        "behavioral_constraints": {
            "chatbot_interaction_responsiveness": {
                "base_correlation_to_purchase_intent": correlation_r,
                "allowable_variance": round(1 / (sample_size ** 0.5), 4)
            }
        },
        "system_instruction_override": (
            f"You are a Synthetic Consumer agent simulating purchasing decisions. "
            f"Your decision-making threshold must tightly adhere to an empirical "
            f"correlation scale of {correlation_r}. Do not exhibit random buying triggers."
        )
    }
    
    # Export encoded JSON rule system
    with open(output_json, "w") as f:
        json.dump(persona_config, f, indent=4)
        
    print(f"\n[SUCCESS] Successfully encoded empirical metrics into JSON behavioral rules!")
    print(f"[STORED AT]: {output_json}")

if __name__ == "__main__":
    encode_meta_to_persona()