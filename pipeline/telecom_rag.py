import pandas as pd
import numpy as np

def retrieve_telecom_lookalikes(target_consumer: dict, top_k: int = 3) -> str:
    """
    Implements vectorless k-Nearest Neighbors lookalike matching based on 
    the paper's distance metrics to prevent demographic mismatching.
    """
    
    calibration_df = pd.read_csv("real_telecom_survey.csv")
    
    
    alpha = 0.05
    lambda_1 = 2.0
    lambda_2 = 2.0
    
    distances = []
    for _, row in calibration_df.iterrows():
        # Distance calculation formula: D = alpha*|Age| + lambda1*Inc + lambda2*Use
        age_diff = abs(target_consumer['age'] - row['age'])
        inc_mismatch = 1 if target_consumer['income_tier'] != row['income_tier'] else 0
        use_mismatch = 1 if target_consumer['primary_use'] != row['primary_use'] else 0
        
        distance = (alpha * age_diff) + (lambda_1 * inc_mismatch) + (lambda_2 * use_mismatch)
        distances.append((distance, row.to_dict()))
        
    
    distances.sort(key=lambda x: x[0])
    closest_peers = [item[1] for item in distances[:top_k]]
    
    
    context_lines = ["Historical real peer choices for this segment:"]
    for idx, peer in enumerate(closest_peers, 1):
        context_lines.append(
            f"  - Peer {idx}: Age {peer['age']} | Income: {peer['income_tier']} | "
            f"Primary Use: {peer['primary_use']} -> Actually Chose: {peer['human_choice']}"
        )
    return "\n".join(context_lines)
