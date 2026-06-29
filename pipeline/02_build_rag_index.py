
import pandas as pd
import numpy as np
import json, os
from pathlib import Path
from rich.console import Console
from tqdm import tqdm

console = Console()
DATA_DIR   = Path("data")
INDEX_DIR  = DATA_DIR / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ── Convert persona vector row → natural language document ────────────────
def persona_to_text(row: dict) -> str:
    """
    Converts a numeric persona vector into a rich text description.
    This is what gets embedded into the RAG index and retrieved
    to ground synthetic consumer responses.
    """
    price_desc = (
        "very price-sensitive and value-driven" if row.get("price_elasticity", 0.5) > 0.7
        else "moderately price-conscious" if row.get("price_elasticity", 0.5) > 0.4
        else "relatively price-insensitive and willing to pay premium"
    )
    brand_desc = (
        "strongly brand-loyal, sticks to known brands" if row.get("brand_loyalty", 0.5) > 0.7
        else "somewhat brand-aware" if row.get("brand_loyalty", 0.5) > 0.4
        else "brand-agnostic, open to new or generic brands"
    )
    digital_desc = (
        "highly digital-native, prefers online shopping" if row.get("digital_affinity", 0.5) > 0.7
        else "comfortable with both online and offline" if row.get("digital_affinity", 0.5) > 0.4
        else "prefers traditional in-store shopping"
    )
    freq_desc = (
        "frequent buyer who shops regularly" if row.get("purchase_frequency", 0.5) > 0.7
        else "moderate shopping frequency" if row.get("purchase_frequency", 0.5) > 0.4
        else "infrequent buyer, deliberate purchases only"
    )
    income_desc = (
        "high-income household" if row.get("income_proxy", 0.5) > 0.7
        else "middle-income household" if row.get("income_proxy", 0.5) > 0.4
        else "budget-constrained household"
    )

    country  = row.get("country", "Unknown")
    age_band = row.get("age_band", "Unknown")
    source   = row.get("source", "")
    sat      = row.get("avg_satisfaction", 0.6)
    sat_desc = "high" if sat > 0.7 else "moderate" if sat > 0.4 else "low"

    text = (
        f"Consumer profile from {country} ({age_band} generation, data source: {source}). "
        f"This consumer is {price_desc}. "
        f"They are {brand_desc}. "
        f"Shopping channel: {digital_desc}. "
        f"Purchase behaviour: {freq_desc}. "
        f"Financial context: {income_desc}. "
        f"Overall satisfaction with past purchases is {sat_desc}. "
        f"Composite behavioural score: {row.get('composite_score', 0.5):.2f}/1.00."
    )
    return text


# ── Build FAISS index ─────────────────────────────────────────────────────
def build_index(persona_csv: str = "data/persona_vectors.csv",
                sample_size: int = 500):
    """
    Build a FAISS vector index from persona documents.
    sample_size: how many personas to index (500 is fast, use all for full run)
    """
    console.print("\n[bold cyan]Building RAG Index...[/]")

    # Load personas
    df = pd.read_csv(persona_csv)
    console.print(f"  Loaded {len(df):,} persona vectors")

    # Sample for speed (increase sample_size for final run)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        console.print(f"  Sampled {sample_size} for indexing")

    # Convert to text documents
    console.print("  Converting to text documents...")
    documents = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Persona→Text"):
        text = persona_to_text(row.to_dict())
        documents.append({
            "id":       str(row.get("CustomerID", _)),
            "text":     text,
            "metadata": {
                "country":           str(row.get("country", "Unknown")),
                "age_band":          str(row.get("age_band", "Unknown")),
                "source":            str(row.get("source", "unknown")),
                "price_elasticity":  float(row.get("price_elasticity", 0.5)),
                "brand_loyalty":     float(row.get("brand_loyalty", 0.5)),
                "digital_affinity":  float(row.get("digital_affinity", 0.5)),
                "composite_score":   float(row.get("composite_score", 0.5)),
            }
        })

    # Save documents as JSONL (backup & inspection)
    docs_path = INDEX_DIR / "documents.jsonl"
    with open(docs_path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    console.print(f"  [green]✓ Documents saved → {docs_path}[/]")

    # Build embeddings using local sentence-transformers
    console.print("  Loading embedding model (sentence-transformers, local)...")
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("all-MiniLM-L6-v2")   # ~80MB, downloads once
    texts = [d["text"] for d in documents]

    console.print("  Embedding documents...")
    embeddings = model.encode(texts, show_progress_bar=True,
                              batch_size=32, normalize_embeddings=True)

    # Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)    # Inner product = cosine similarity (normalized vecs)
    index.add(embeddings.astype(np.float32))

    # Save index
    faiss.write_index(index, str(INDEX_DIR / "personas.index"))
    console.print(f"  [green]✓ FAISS index saved → {INDEX_DIR}/personas.index[/]")
    console.print(f"  Index size: {index.ntotal} vectors, dim={dim}")

    return index, documents, model



_FAISS_INDEX = None
_EMBED_MODEL = None
_DOC_CACHE = None

import pandas as pd
import numpy as np

def retrieve_telecom_lookalikes(target_consumer: dict, top_k: int = 3) -> str:
    """
    Implements vectorless k-Nearest Neighbors lookalike matching based on 
    the paper's distance metrics to prevent demographic mismatching.
    """
    calibration_df = pd.read_csv("real_telecom_survey.csv")
    
    # Weights for distance penalties
    alpha = 0.05
    lambda_1 = 2.0
    lambda_2 = 2.0
    
    distances = []
    for _, row in calibration_df.iterrows():
        # Calculate penalty metric
        age_diff = abs(target_consumer['age'] - row['age'])
        inc_mismatch = 1 if target_consumer['income_tier'] != row['income_tier'] else 0
        use_mismatch = 1 if target_consumer['primary_use'] != row['primary_use'] else 0
        
        distance = (alpha * age_diff) + (lambda_1 * inc_mismatch) + (lambda_2 * use_mismatch)
        distances.append((distance, row.to_dict()))
        
    # Sort and pick top nearest lookalikes
    distances.sort(key=lambda x: x[0])
    closest_peers = [item[1] for item in distances[:top_k]]
    
    # Format directly into prompt context
    context_lines = ["Historical real peer choices for this segment:"]
    for idx, peer in enumerate(closest_peers, 1):
        context_lines.append(
            f"  - Peer {idx}: Age {peer['age']} | Income: {peer['income_tier']} | "
            f"Primary Use: {peer['primary_use']} -> Actually Chose: {peer['human_choice']}"
        )
    return "\n".join(context_lines)