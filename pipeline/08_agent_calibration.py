import json
import numpy as np
import pandas as pd
from pathlib import Path

def run_agent_calibration_harness():
    print("\n=== Initializing Digital Twin Calibration Harness ===")
    ROOT_DIR = Path(__file__).parent.parent
    
    # Paths
    rules_json = ROOT_DIR / "outputs" / "encoded_agent_persona_rules.json"
    output_csv = ROOT_DIR / "outputs" / "calibration_drift_results.csv"
    
    if not rules_json.exists():
        print(f"[ERROR] Persona rules missing. Run module 07 first.")
        return
        
    with open(rules_json, "r") as f:
        persona = json.load(f)
        
    target_r = persona["behavioral_constraints"]["chatbot_interaction_responsiveness"]["base_correlation_to_purchase_intent"]
    print(f"[TARGET HUMAN BASELINE]: r = {target_r}")
    
    # --- SIMULATING AGENT RESPONSES (The Choice Stimulus Engine) ---
    # We simulate 100 interaction scenarios to evaluate the agent's choice logic
    np.random.seed(42)
    scenarios = 100
    
    # Simulating the Agent's perceived Chatbot Responsiveness score (Scale 1-7)
    agent_perceived_responsiveness = np.random.uniform(1, 7, scenarios)
    
    # Introduce an uncalibrated drift factor (simulating a standard, raw LLM bias)
    llm_drift_noise = np.random.normal(0, 2.5, scenarios) 
    
    # Calculate simulated Agent Purchase Intent based on its internal prompt weight
    agent_purchase_intent = (target_r * agent_perceived_responsiveness) + llm_drift_noise
    agent_purchase_intent = np.clip(agent_purchase_intent, 1, 7) # Keep on 1-7 Likert scale
    
    # --- CALCULATING THE DRIFT GAP (Beta_agent - Beta_human) ---
    # Calculate the agent's simulated correlation coefficient
    agent_matrix = np.corrcoef(agent_perceived_responsiveness, agent_purchase_intent)
    agent_r = round(agent_matrix[0, 1], 4)
    
    # Mathematical Drift Gap
    drift_gap = round(agent_r - target_r, 4)
    
    print(f"[SIMULATED AGENT RESPONSE]: r = {agent_r}")
    print(f"[CRITICAL DRIFT GAP DETECTED]: {drift_gap}")
    
    # Save the calibration run data
    calibration_df = pd.DataFrame({
        "Scenario_ID": range(1, scenarios + 1),
        "Perceived_Responsiveness": agent_perceived_responsiveness,
        "Simulated_Purchase_Intent": agent_purchase_intent,
        "Target_Human_r": [target_r] * scenarios,
        "Agent_Calculated_r": [agent_r] * scenarios,
        "Drift_Gap": [drift_gap] * scenarios
    })
    
    calibration_df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Calibration run complete. Results saved permanently.")
    print(f"[STORED AT]: {output_csv}")
    
    if abs(drift_gap) > 0.1:
        print("⚠️ [STATUS]: UNCALIBRATED. Prompt optimization tuning required to close the gap.")
    else:
        print("✅ [STATUS]: CALIBRATED. Digital Twin matches human baseline boundaries.")

if __name__ == "__main__":
    run_agent_calibration_harness()