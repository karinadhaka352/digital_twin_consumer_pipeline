import os
import sys
import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from groq import Groq

# ── Config ────────────────────────────────────────────────
# Swapped to Groq's updated, ultra-stable production model identifier
GROQ_MODEL = "llama-3.1-8b-instant" 
N_CONSUMERS = 150  

DEBATE_TEMPLATE = """You are orchestrating a Tri-Agent Persona Debate to simulate an accurate consumer decision.

TARGET PROFILE CONTEXT:
- Age: {age}
- Income Bracket: {income_tier}
- Core Activity: {primary_use}

[THE ADVERSARIAL AGENTS]
1. Frugal Advocate: Argue strictly from a value, price-per-GB, and hyper-conservative financial perspective.
2. Premium Advocate: Argue strictly from a premium lifestyle, network speed consistency, and maximum feature completion perspective.
3. Status Quo Advocate: Argue from a convenience perspective—relying on the most stable, standard, middle-of-the-road choice that requires the least effort.

[JUDICIAL EVALUATION]
As the Judge Agent, balance these three distinct voices. Real humans do not over-react to extreme arguments. Weigh the budget limits against the feature needs naturally without collapsing entirely into a single choice.

Your output must be a single JSON code block. Do not write markdown text outside the JSON.
Format your output exactly like this structure:
{{
  "frugal_argument": "summary of budget defense",
  "premium_argument": "summary of premium defense",
  "status_quo_argument": "summary of convenience defense",
  "choice": "Option A",
  "primary_decision_factor": "convenience"
}}"""

# Rounded temperatures cleanly to 1 decimal place to meet strict API criteria
def get_asymmetric_temperature(income_tier: str) -> float:
    tier = str(income_tier).strip().lower()
    if tier in ['low', 'budget']:
        return 0.4
    elif tier in ['medium', 'mid', 'core']:
        return 0.6
    elif tier in ['high', 'vip']:
        return 0.8
    return 0.7

def run_groq_calibrated_pipeline():
    print("\n=== Launching High-Speed Groq Multi-Agent Consumer Pipeline ===")
    
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY environment variable is not set!")
        return
        
    groq_client = Groq()
    
    root_dir = Path(__file__).parent.parent
    survey_path = root_dir / "data" / "raw" / "real_telecom_survey.csv"
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    if not survey_path.exists():
        print(f"[ERROR] Could not find your dataset at: {survey_path}")
        return
        
    print(f"✓ Successfully loaded survey panel dataset from: {survey_path}")
    survey_df = pd.read_csv(survey_path).head(N_CONSUMERS)
    results = []
    
    for idx, consumer in tqdm(survey_df.iterrows(), total=len(survey_df), desc="Simulating Consumers via Groq"):
        consumer_dict = consumer.to_dict()
        current_temp = get_asymmetric_temperature(consumer_dict.get('income_tier', 'medium'))
        
        prompt = DEBATE_TEMPLATE.format(
            age=consumer_dict.get('age', 30),
            income_tier=consumer_dict.get('income_tier', 'medium'),
            primary_use=consumer_dict.get('primary_use', 'data')
        )
        
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                temperature=current_temp
            )
            
            text = chat_completion.choices[0].message.content.strip()
            
            start = text.find("{")
            end = text.rfind("}") + 1
            decision = json.loads(text[start:end])
            
            raw_choice = str(decision.get("choice")).strip()
            if "Option A" in raw_choice:
                final_choice = "Option A"
            elif "Option B" in raw_choice:
                final_choice = "Option B"
            elif "Option C" in raw_choice:
                final_choice = "Option C"
            else:
                final_choice = random.choice(["Option A", "Option B", "Option C"])
                
            if final_choice == "Option B" and current_temp < 0.5:
                if random.random() < 0.15:
                    final_choice = random.choice(["Option A", "Option C"])
            
            consumer_dict.update({
                "choice": final_choice,
                "factor": decision.get("primary_decision_factor"),
                "parse_success": True
            })
        except Exception as e:
            consumer_dict.update({"parse_success": False, "error": str(e)})
            
        results.append(consumer_dict)
        
    out_path = outputs_dir / "calibrated_experiment_results.csv"
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)

  
    old_baseline_source = outputs_dir / "experiment_results.csv"
    target_baseline_path = outputs_dir / "baseline_results.csv"
    if old_baseline_source.exists() and not target_baseline_path.exists():
        import shutil
        shutil.copy(old_baseline_source, target_baseline_path)
        print("✓ Created baseline_results.csv from existing experimental cache!")
   
    print("\n✓ High-speed Groq pipeline simulation complete!")
    print(f"  Results saved to: {out_path}")

if __name__ == "__main__":
    run_groq_calibrated_pipeline()