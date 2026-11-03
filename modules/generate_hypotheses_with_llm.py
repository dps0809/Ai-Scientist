import os
import json
import requests
# pyrefly: ignore [missing-import]
from .config import OLLAMA_MODEL, OLLAMA_URL, OUTPUT_DIR


def generate_hypotheses_with_llm(context):
    print("\n" + "=" * 70)
    print("STAGE 3: LOCAL LLM HYPOTHESIS GENERATION")
    print("=" * 70)

    target = context.get("target", "Outcome")
    all_features = context.get("all_features", [])

    stage1_deps = context.get("stage_1_individual_dependencies", [])
    stage2_combs = context.get("stage_2_feature_combinations", [])
    stage2_inters = context.get("stage_2_feature_interactions", [])
    inter_corr = context.get("inter_variable_correlations", [])
    summary_stats = context.get("summary_statistics", {})

    stage1_text = json.dumps(stage1_deps[:8], indent=2)
    stage2_combs_text = json.dumps(stage2_combs[:8], indent=2)
    stage2_inters_text = json.dumps(stage2_inters[:6], indent=2)
    inter_corr_text = json.dumps(inter_corr[:8], indent=2)

    prompt = f"""You are an elite scientific hypothesis generation assistant specializing in biomedical and tabular data discovery.

A quantitative discovery pipeline analyzed a dataset ({context.get('dataset_rows', '?')} rows, {context.get('dataset_columns', '?')} columns).
Target variable: {target}
Features: {', '.join(all_features)}

=== EVIDENCE SECTION 1: STAGE 1 INDIVIDUAL FEATURE DEPENDENCY ANALYSIS ===
Calculated independent associations between each feature and {target} (Pearson r, Spearman rho, Mutual Information, Significance):
{stage1_text}

=== EVIDENCE SECTION 2: STAGE 2 FEATURE COMBINATION DISCOVERY (JOINT PREDICTIVE SYNERGY) ===
Calculated Joint Mutual Information and 5-Fold Cross-Validated Model Predictive Synergy (Single scores vs Combined score, Synergy Gain):
{stage2_combs_text}

=== EVIDENCE SECTION 3: STAGE 2 FEATURE INTERACTION ANALYSIS (MODERATION & PRODUCT TERMS) ===
Evaluated whether the effect of one feature varies across levels of another feature (Additive model vs Interaction model X_A * X_B):
{stage2_inters_text}

=== EVIDENCE SECTION 4: INTER-VARIABLE CORRELATIONS (POTENTIAL CONFOUNDING / MEDIATION) ===
Correlations between predictor variables:
{inter_corr_text}

Generate exactly 7 high-quality, testable scientific hypotheses adhering to this taxonomy:

1. (Primary Individual Dependency): State the directional relationship of the #1 ranked feature with {target}, citing its Pearson r, Spearman rho, and Mutual Information.
2. (Secondary Individual Dependency): State the directional relationship of another top-ranked feature with {target}, citing its metrics.
3. (Primary Feature Combination Synergy): Propose that combining two specific features (from Section 2) provides greater predictive information about {target} than either feature alone. Cite their individual scores, combined score, synergy gain, and Joint MI.
4. (Secondary Feature Combination Synergy): Propose that combining two specific features (from Section 2) provides greater predictive information about {target} than either feature alone. Cite their individual scores, combined score, synergy gain, and Joint MI.
5. (Primary Feature Interaction / Moderation): Propose that the relationship between a feature and {target} varies across levels of another feature (from Section 3). Cite the interaction delta and p-value.
6. (Secondary Feature Interaction / Moderation): Propose that the relationship between a feature and {target} varies across levels of another feature (from Section 3). Cite the interaction delta and p-value.
7. (Comparative or Confounding Relationship): Compare the relative importance of two correlated predictors (from Section 4) or propose a confounding/mediating mechanism affecting {target}.

STRICT FORMATTING REQUIREMENTS:
- Return ONLY valid JSON, starting immediately with '[' and ending with ']'.
- No markdown formatting around the JSON, no commentary.
- Every hypothesis MUST cite the exact numbers from the evidence sections above.
- Every proposed_experiment must describe a specific statistical or machine learning test to run on the dataset.

JSON Structure:
[
  {{
    "hypothesis_id": "H1",
    "hypothesis_type": "individual_dependency",
    "hypothesis": "Clear statement of the hypothesis.",
    "evidence_from_data": "Exact statistics cited from the evidence sections.",
    "proposed_experiment": "Specific statistical test, cross-validation, or controlled model to execute."
  }}
]
"""

    is_qwen_reasoning = "qwen" in OLLAMA_MODEL.lower()

    if is_qwen_reasoning:
        # Pre-close the <think> block and pre-fill the opening '[' to bypass
        # the model's lengthy chain-of-thought monologue, reducing generation
        # time from ~3 minutes to ~20 seconds on CPU.
        formatted_prompt = f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n[\n"
        prefix = "[\n"
        use_raw = True
    else:
        formatted_prompt = prompt
        prefix = ""
        use_raw = False

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": formatted_prompt,
        "raw": use_raw,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_predict": 6000
        }
    }

    text = ""
    try:
        print("\nSending discovered evidence to local LLM...")
        print("Waiting for LLM response...")

        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600)
        response.raise_for_status()

        if payload.get("stream", False):
            print("\nStreaming LLM response:\n")
            if prefix:
                print(prefix, end="", flush=True)
            text_chunks = [prefix] if prefix else []
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    response_chunk = chunk.get("response", "")
                    if response_chunk:
                        print(response_chunk, end="", flush=True)
                        text_chunks.append(response_chunk)
            print()
            text = "".join(text_chunks).strip()
        else:
            result = response.json()
            text = prefix + result.get("response", "").strip()
            print("\nRaw LLM response:")
            print(text)

        if not text:
            print("\nLLM returned an empty response.")
            return []

        # Extract the JSON array robustly
        hypotheses = _extract_json_array(text)

        if hypotheses is None:
            print("\n[ERROR] Could not extract a valid JSON array from LLM output.")
            return []

        if not isinstance(hypotheses, list):
            print("\nLLM response was not a JSON list.")
            return []

        print("\n" + "=" * 70)
        print("GENERATED HYPOTHESES")
        print("=" * 70)

        for hypothesis in hypotheses:
            print(f"\n{hypothesis.get('hypothesis_id', 'H?')} [{hypothesis.get('hypothesis_type', 'Hypothesis')}]")
            print("-" * 70)
            print("Hypothesis:")
            print(hypothesis.get("hypothesis", "N/A"))
            print("\nEvidence:")
            print(hypothesis.get("evidence_from_data", "N/A"))
            print("\nProposed Experiment:")
            print(hypothesis.get("proposed_experiment", "N/A"))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "generated_hypotheses.json")
        with open(output_path, "w") as f:
            json.dump(hypotheses, f, indent=4)

        print("\nHypotheses saved successfully:")
        print(output_path)
        return hypotheses

    except requests.exceptions.Timeout:
        print("\nLLM generation timed out.")
        print("Ollama is running, but the model did not respond within the timeout.")
        return []
    except requests.exceptions.ConnectionError:
        print("\nCould not connect to Ollama.")
        print(f"Make sure Ollama is running and '{OLLAMA_MODEL}' is installed.")
        return []
    except Exception as e:
        print("\nLLM generation failed:")
        print(e)
        return []


def _extract_json_array(text: str):
    """
    Find the LAST complete JSON array in *text* and return the parsed list.
    Returns None if no valid array is found.
    """
    text = text.replace("```json", "").replace("```", "")
    last_close = text.rfind("]")
    if last_close == -1:
        return None

    depth = 0
    in_string = False
    i = last_close
    while i >= 0:
        ch = text[i]
        if i > 0 and text[i - 1] == "\\" and in_string:
            i -= 1
            continue
        if ch == '"':
            in_string = not in_string
        if not in_string:
            if ch == "]":
                depth += 1
            elif ch == "[":
                depth -= 1
                if depth == 0:
                    candidate = text[i:last_close + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        i -= 1
    return None
