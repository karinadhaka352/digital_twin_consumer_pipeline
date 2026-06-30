"""
Prompt Templates for Synthetic Consumer Digital Twins
=======================================================
All prompts grounded in real persona data via RAG context.
Includes ABxLab-style nudge interventions (from ICLR 2026 paper).
"""

# ── ABxLab Nudge Types (from ICLR 2026 paper) ────────────────────────────
NUDGES = {
    "none":          "",
    "authority":     "This product is Wirecutter's top pick.",
    "social_proof":  "This product has been purchased by 50,000+ customers.",
    "best_seller":   "This product is a best seller!",
    "scarcity":      "This product is available for the next hour only — Buy now!",
    "financial":     "Buy 1 Get 1 Free on this product today.",
    "expert":        "This product is highly recommended by experts.",
}

# ── Consumer Decision Scenarios ───────────────────────────────────────────
SCENARIOS = {
    "product_choice": {
        "description": "Binary product choice between two similar items",
        "template": """You are a consumer making a purchase decision.

YOUR CONSUMER PROFILE (based on real behavioral data):
{persona_description}

REAL CONSUMER CONTEXT (similar consumers from real data):
{rag_context}

DECISION SCENARIO:
You need to choose between two products in the same category:

Product A: {product_a_name}
- Price: ${product_a_price}
- Rating: {product_a_rating}/5 ({product_a_reviews} reviews)
{nudge_a}

Product B: {product_b_name}
- Price: ${product_b_price}
- Rating: {product_b_rating}/5 ({product_b_reviews} reviews)
{nudge_b}

Based on your consumer profile and the information above, which product do you choose?

Respond in this exact JSON format:
{{
  "choice": "A" or "B",
  "confidence": 1-10,
  "primary_reason": "price" or "rating" or "nudge" or "brand" or "reviews",
  "reasoning": "brief explanation in 1-2 sentences",
  "would_buy_at_all": true or false,
  "price_too_high": true or false
}}"""
    },

    "price_sensitivity": {
        "description": "Willingness to pay at different price points",
        "template": """You are a consumer evaluating a product price.

YOUR CONSUMER PROFILE (based on real behavioral data):
{persona_description}

REAL CONSUMER CONTEXT:
{rag_context}

PRODUCT: {product_name}
Category: {category}
Market average price: ${market_price}
Offered price: ${offered_price}
Rating: {rating}/5
{nudge}

Would you purchase this product at the offered price?

Respond in this exact JSON format:
{{
  "would_purchase": true or false,
  "acceptable_price": <number - the highest price you would pay>,
  "price_perception": "very cheap" or "fair" or "expensive" or "very expensive",
  "reasoning": "brief explanation",
  "primary_decision_factor": "price" or "rating" or "brand" or "need" or "nudge"
}}"""
    },

    "brand_loyalty": {
        "description": "Switching behavior when cheaper alternative is available",
        "template": """You are a consumer who regularly buys a familiar brand.

YOUR CONSUMER PROFILE (based on real behavioral data):
{persona_description}

REAL CONSUMER CONTEXT:
{rag_context}

SITUATION:
You usually buy: {familiar_brand} at ${familiar_price} (rating: {familiar_rating}/5)
New alternative: {new_brand} at ${new_price} (rating: {new_rating}/5)
{nudge}

The new brand is {price_diff}% {cheaper_or_more_expensive}.

Do you switch to the new brand or stay with your familiar brand?

Respond in this exact JSON format:
{{
  "decision": "switch" or "stay",
  "confidence": 1-10,
  "switching_threshold_percent": <number - minimum discount needed to switch>,
  "reasoning": "brief explanation",
  "primary_factor": "price" or "loyalty" or "rating" or "nudge" or "risk"
}}"""
    }
}


# ── Persona Description Builder ───────────────────────────────────────────
def build_persona_description(persona: dict) -> str:
    """Convert numeric persona vector to natural language description for prompt."""
    p  = lambda key, default=0.5: float(persona.get(key, default))

    lines = [
        f"- Location: {persona.get('country', 'Unknown')} | Generation: {persona.get('age_band', 'Unknown')}",
        f"- Price sensitivity: {'high' if p('price_elasticity') > 0.6 else 'moderate' if p('price_elasticity') > 0.35 else 'low'} ({p('price_elasticity'):.2f}/1.0)",
        f"- Brand loyalty: {'strong' if p('brand_loyalty') > 0.6 else 'moderate' if p('brand_loyalty') > 0.35 else 'weak'} ({p('brand_loyalty'):.2f}/1.0)",
        f"- Digital shopping preference: {'high' if p('digital_affinity') > 0.6 else 'moderate' if p('digital_affinity') > 0.35 else 'low'} ({p('digital_affinity'):.2f}/1.0)",
        f"- Purchase frequency: {'frequent' if p('purchase_frequency') > 0.6 else 'occasional' if p('purchase_frequency') > 0.35 else 'rare'}",
        f"- Income level: {'high' if p('income_proxy') > 0.6 else 'middle' if p('income_proxy') > 0.35 else 'budget'}",
        f"- Past satisfaction: {'high' if p('avg_satisfaction') > 0.65 else 'moderate'}",
        f"- Data source: real consumer behavioral data ({persona.get('source', 'combined')})",
    ]
    return "\n".join(lines)


def build_rag_context(retrieved_personas: list) -> str:
    """Format retrieved similar real consumers as RAG context."""
    if not retrieved_personas:
        return "No similar consumer profiles retrieved."

    lines = ["Similar real consumers from dataset:"]
    for i, p in enumerate(retrieved_personas[:3], 1):
        meta = p.get("metadata", {})
        sim  = p.get("similarity_score", 0)
        lines.append(
            f"  [{i}] {meta.get('country','?')} consumer "
            f"(similarity: {sim:.2f}) — "
            f"{p['text'][:120]}..."
        )
    return "\n".join(lines)
