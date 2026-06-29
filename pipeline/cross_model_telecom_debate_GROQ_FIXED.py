import os, sys, json, time
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR  = Path(r"C:\Users\summer.intern12\Downloads\digital_twin_consumer_pipeline")
DOWNLOADS = Path.home() / "Downloads"

sys.path.append(str(ROOT_DIR / "pipeline"))
from telecom_rag import retrieve_telecom_lookalikes

# Only currently-supported Groq models. Verify at https://console.groq.com/docs/models
# if you hit another "model_decommissioned" error in the future.
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

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
    if tier in ["low", "budget"]:
        return 0.2
    elif tier in ["medium", "mid", "core"]:
        return 0.4
    elif tier in ["high", "vip"]:
        return 0.65
    return 0.5


def _parse_json(text: str):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None


def call_groq(client, model, prompt, temperature, retries=2):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=300,
            )
            return resp.choices[0].message.content, None
        except Exception as e:
            err_str = str(e)
            if "model_decommissioned" in err_str or "does not exist" in err_str.lower():
                # No point retrying — the model itself is gone.
                return None, f"FATAL (model unavailable): {err_str[:150]}"
            if attempt == retries - 1:
                return None, err_str[:150]
            time.sleep(1.5)
    return None, "unreachable"


def check_key():
    if not os.getenv("GROQ_API_KEY"):
        print("[FATAL] GROQ_API_KEY not found in .env")
        print("Get a free key at: https://console.groq.com/keys")
        sys.exit(1)


def run_cross_model_test(n_consumers: int = 50):
    print("\n=== Cross-Model Telecom Debate Test (Groq, FIXED) ===")
    check_key()

    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    survey_path = None
    for path in [DOWNLOADS / "telecom_survey.csv", ROOT_DIR / "telecom_survey.csv"]:
        if path.exists():
            survey_path = path
            break
    if survey_path is None:
        print("[FATAL] Could not find telecom_survey.csv")
        sys.exit(1)

    survey_df = pd.read_csv(survey_path).head(n_consumers)
    print(f"  Loaded {len(survey_df)} consumers from {survey_path}")
    print(f"  Models to test: {GROQ_MODELS}\n")

    all_results = []
    model_summary = {}

    for model_name in GROQ_MODELS:
        print(f"\u2699 Evaluating Decision Dynamics for Model Architecture: [{model_name}]")
        n_failed = 0
        fatal_error = None

        for idx, consumer in tqdm(survey_df.iterrows(), total=len(survey_df), desc=f"  Running {model_name}"):
            consumer_dict = consumer.to_dict()
            context = retrieve_telecom_lookalikes(consumer_dict, top_k=3)
            temp = get_asymmetric_temperature(consumer_dict["income_tier"])

            prompt = DEBATE_TEMPLATE.format(
                age=consumer_dict["age"],
                income_tier=consumer_dict["income_tier"],
                primary_use=consumer_dict["primary_use"],
                rag_context=context,
            )

            raw, error = call_groq(client, model_name, prompt, temp)
            decision = _parse_json(raw) if raw else None

            row = {
                "model": model_name,
                "age": consumer_dict["age"],
                "income_tier": consumer_dict["income_tier"],
                "primary_use": consumer_dict["primary_use"],
            }
            if decision is not None:
                row.update({
                    "choice": decision.get("choice"),
                    "factor": decision.get("primary_decision_factor"),
                    "parse_success": True,
                })
            else:
                n_failed += 1
                row.update({"choice": None, "factor": None, "parse_success": False, "error": error})
                if error and "FATAL" in error:
                    fatal_error = error
                    tqdm.write(f"    [FATAL] {model_name} appears unavailable: {error}")
                    break  # stop wasting calls on a dead model

            all_results.append(row)
            time.sleep(0.3)

        n_success = len(survey_df) - n_failed
        model_summary[model_name] = {
            "n_success": n_success,
            "n_total": len(survey_df),
            "fatal_error": fatal_error,
        }

        if fatal_error:
            print(f"  [SKIPPED] {model_name}: model unavailable, 0 usable results.\n")
        elif n_success == 0:
            print(f"  [FAILED] {model_name}: 0/{len(survey_df)} successful \u2014 check API key/quota.\n")
        else:
            print(f"  {model_name}: {n_success}/{len(survey_df)} successful ({n_success/len(survey_df):.1%})\n")
            if n_success < 30:
                print(f"  [WARNING] Sample size below 30 \u2014 treat distribution as preliminary.\n")

    out_df = pd.DataFrame(all_results)
    out_path = ROOT_DIR / "outputs" / "cross_model_comparison_results_FIXED.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\u2713 Results saved to: {out_path}")

    # ── Clear, honest summary — only for models that actually produced data ──
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (only models with usable data shown)")
    print("=" * 60)

    any_usable = False
    for model_name, info in model_summary.items():
        if info["fatal_error"]:
            print(f"\n{model_name}: SKIPPED \u2014 model unavailable on Groq (decommissioned or renamed)")
            continue
        sub = out_df[(out_df["model"] == model_name) & (out_df["parse_success"] == True)]
        if len(sub) == 0:
            print(f"\n{model_name}: NO USABLE DATA \u2014 0 successful trials, do not report any distribution for this model")
            continue
        any_usable = True
        dist = sub["choice"].value_counts(normalize=True).round(4)
        warning_text = " — SMALL SAMPLE WARNING" if len(sub) < 30 else ""
        print(f"\n{model_name} (n={len(sub)}){warning_text}:")
        print(dist.to_string())

    if not any_usable:
        print("\n[FATAL] No model in this run produced usable data. Check your GROQ_API_KEY")
        print("and model availability at https://console.groq.com/docs/models before rerunning.")
        sys.exit(1)

    print("\nNext: compare each model's distribution to the human baseline")
    print("(Option A/B/C = 0.55/0.21/0.24) and to your local llama3 result")
    print("(0.65/0.20/0.16, n=150) to see how the skew changes with model scale.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_consumers", type=int, default=50)
    args = parser.parse_args()
    run_cross_model_test(args.n_consumers)
