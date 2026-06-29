import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


sys.path.append(str(Path(__file__).parent))
from telecom_rag import retrieve_telecom_lookalikes

from ollama import Client

MODELS = ["llama3"]
N_CONSUMERS = 15  

DEBATE_TEMPLATE = """You are orchestrating a Tri-Agent Persona Debate to simulate an accurate consumer decision.

TARGET PROFILE CONTEXT:
- Age: {age}
- Income Bracket: {income_tier}
- Core Activity: {primary_use}

REAL HUMAN BASE RATES (LOOKALIKES):
{rag_context}

[STEP 1: ADVERSARIAL ARGUMENTS]
- Frugal Advocate: Argue strictly from a value, savings, and hyper-conservative financial perspective.
- Premium Advocate: Argue strictly from a premium lifestyle, comfort, feature completion, and user-experience perspective.

[STEP 2: JUDICIAL DECISION]
As the Judge Agent, evaluate the opposing arguments above, respect the provided Real Human Base Rates context, and make the final choice prediction.

Respond in this exact JSON format:
{{
  "frugal_argument": "summary of budget defense",
  "premium_argument": "summary of premium defense",
  "choice": "Option A" or "Option B" or "Option C",
  "primary_decision_factor": "price" or "utility" or "experience" or "peer_alignment"
}}"""

def get_asymmetric_temperature(income_tier: str) -> float:
    tier = str(income_tier).strip().lower()
    if tier in ['low', 'budget']:
        return 0.2
    elif tier in ['medium', 'mid', 'core']:
        return 0.4
    elif tier in ['high', 'vip']:
        return 0.65
    return 0.5

def run_calibrated_pipeline():
    print("\n=== Launching Calibrated Multi-Agent Consumer Pipeline ===")
    
    
    local_client = Client(host='http://localhost:11434')
    
    
    downloads_dir = Path(r"C:\Users\summer.intern12\Downloads")
    root_dir = Path(r"C:\Users\summer.intern12\Downloads\digital_twin_consumer_pipeline")
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
   
    survey_df = None
    possible_survey_paths = [
        downloads_dir / "telecom_survey.csv",
        downloads_dir / "telecom_survey.csv.csv",
        downloads_dir / "telecom_survey",
        downloads_dir / "telecom_survey.txt",
        root_dir / "telecom_survey.csv",
        root_dir / "telecom_survey.csv.csv",
        root_dir / "telecom_survey"
    ]
    
    for path in possible_survey_paths:
        if path.exists():
            try:
                survey_df = pd.read_csv(path).head(N_CONSUMERS)
                print(f"✓ Successfully loaded survey panel dataset from: {path}")
                break
            except:
                pass
                
    if survey_df is None:
        print("\n[ERROR] Could not find your 'telecom_survey.csv' file anywhere in Downloads or your project folder.")
        print("Please verify the file name inside your Downloads directory.")
        return
        
    results = []
    
    for idx, consumer in tqdm(survey_df.iterrows(), total=len(survey_df), desc="Simulating Consumers"):
        consumer_dict = consumer.to_dict()
        context = retrieve_telecom_lookalikes(consumer_dict, top_k=3)
        current_temp = get_asymmetric_temperature(consumer_dict['income_tier'])
        
        prompt = DEBATE_TEMPLATE.format(
            age=consumer_dict['age'],
            income_tier=consumer_dict['income_tier'],
            primary_use=consumer_dict['primary_use'],
            rag_context=context
        )
        
        try:
            response = local_client.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": current_temp}
            )
            text = response["message"]["content"].strip()
            
            start = text.find("{")
            end = text.rfind("}") + 1
            decision = json.loads(text[start:end])
            
            consumer_dict.update({
                "choice": decision.get("choice"),
                "factor": decision.get("primary_decision_factor"),
                "parse_success": True
            })
        except Exception as e:
            consumer_dict.update({"parse_success": False, "error": str(e)})
            
        results.append(consumer_dict)
        
    
    out_path = outputs_dir / "calibrated_experiment_results.csv"
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    
    print("\n✓ Pipeline simulation complete!")
    print(f"  Results cleanly saved to: {out_path}")

if __name__ == "__main__":
    run_calibrated_pipeline()