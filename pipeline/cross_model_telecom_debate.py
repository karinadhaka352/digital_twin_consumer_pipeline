import os
import sys
import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from groq import Groq

# ── Config ────────────────────────────────────────────────
# Swapped to Groq's active production 8B and 70B models
MODELS_TO_TEST = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
N_CONSUMERS = 50  

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

Your output must be a single raw JSON block. Do not write markdown text outside the JSON.
Format your output exactly like this structure:
{{
  "frugal_argument": "summary of budget defense",
  "premium_argument": "summary of premium defense",
  "status_quo_argument": "summary of convenience defense",
  "choice": "Option A",
  "primary_decision_factor": "convenience"
}}"""

def get_asymmetric_temperature(income_tier: str) -> float:
    tier = str(income_tier).strip().lower()
    if tier in ['low', 'budget']: return 0.4
    elif tier in ['medium', 'mid', 'core']: return 0.6
    elif tier in ['high', 'vip']: return 0.8
    return 0.7

def main():
    print("\n=== Launching Cross-Model Telecom Debate Evaluation Matrix ===")
    
    if not os.environ.get("GROQ_API_KEY"):
        print("[FATAL] No GROQ_API_KEY found in environment variables!")
        sys.exit(1)
        
    groq_client = Groq()
    ROOT_DIR = Path(__file__).parent.parent
    survey_path = ROOT_DIR / "data" / "raw" / "real_telecom_survey.csv"
    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    if not survey_path.exists():
        print(f"[ERROR] Dataset missing at: {survey_path}")
        return
        
    survey_df = pd.read_csv(survey_path).head(N_CONSUMERS)
    all_model_results = []

    for model_name in MODELS_TO_TEST:
        print(f"\n🚀 Evaluating Decision Dynamics for Model Architecture: [{model_name}]")
        
        for idx, consumer in tqdm(survey_df.iterrows(), total=len(survey_df), desc=f"Running {model_name}"):
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
                    model=model_name,
                    temperature=current_temp
                )
                text = chat_completion.choices[0].message.content.strip()
                
                start = text.find("{")
                end = text.rfind("}") + 1
                
                if start == -1 or end == 0:
                    raise ValueError(f"No JSON block braces found in response.")
                    
                decision = json.loads(text[start:end])
                
                raw_choice = str(decision.get("choice")).strip()
                if "Option A" in raw_choice: final_choice = "Option A"
                elif "Option B" in raw_choice: final_choice = "Option B"
                elif "Option C" in raw_choice: final_choice = "Option C"
                else: final_choice = random.choice(["Option A", "Option B", "Option C"])
                
                all_model_results.append({
                    "model_tested": model_name,
                    "consumer_idx": idx,
                    "income_tier": consumer_dict.get('income_tier'),
                    "simulated_choice": final_choice,
                    "primary_factor": decision.get("primary_decision_factor"),
                    "parse_success": True
                })
            except Exception as e:
                all_model_results.append({
                    "model_tested": model_name,
                    "consumer_idx": idx,
                    "simulated_choice": "Failed",
                    "parse_success": False,
                    "error": str(e)
                })

    out_df = pd.DataFrame(all_model_results)
    summary_path = outputs_dir / "cross_model_comparison_results.csv"
    out_df.to_csv(summary_path, index=False)
    print(f"\n✓ Cross-model matrix benchmarking finished! Matrix saved to: {summary_path}")

    print("\n================== BENCHMARK SUMMARY ==================")
    for model_name in MODELS_TO_TEST:
        m_df = out_df[(out_df["model_tested"] == model_name) & (out_df["parse_success"] == True)]
        if len(m_df) > 0:
            counts = m_df["simulated_choice"].value_counts(normalize=True) * 100
            print(f"\nDistribution breakdown for [{model_name}]:")
            for opt in ["Option A", "Option B", "Option C"]:
                print(f"  {opt}: {counts.get(opt, 0.0):.1f}%")
        else:
            print(f"\n[ERROR] No successful simulation data parsed for model: {model_name}")

if __name__ == "__main__":
    main()