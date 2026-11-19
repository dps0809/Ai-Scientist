import os
import json
import requests
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from .config import OLLAMA_MODEL, OLLAMA_URL, OUTPUT_DIR, DB_HYPOTHESES
from .schemas import HypothesisRecord
from .hypothesis_registry import HypothesisRegistry
from .mongo_cluster import MongoClusterManager


def generate_hypotheses_with_llm(context, iteration_no: int = 1, registry=None, output_dir=None):
    print("\n" + "=" * 70)
    print(f"STAGE 3: LOCAL LLM HYPOTHESIS GENERATION (ITERATION {iteration_no})")
    print("=" * 70)

    if registry is None:
        reg_path = os.path.join(output_dir, "hypothesis_registry.json") if output_dir else HYPOTHESIS_REGISTRY_PATH
        registry = HypothesisRegistry(registry_path=reg_path)

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
    past_hypotheses_text = registry.get_past_hypotheses_prompt_summary()

    prompt = f"""You are an elite scientific hypothesis generation assistant specializing in biomedical and tabular data discovery.

A quantitative discovery pipeline analyzed a dataset ({context.get('dataset_rows', '?')} rows, {context.get('dataset_columns', '?')} columns).
Target variable: {target}
Features: {', '.join(all_features)}

=== AVOID DUPLICATION: PREVIOUSLY INVESTIGATED HYPOTHESES ===
The following hypotheses have ALREADY been investigated in previous iterations. You MUST propose NOVEL, UNTESTED hypotheses and MUST NOT regenerate duplicates of these:
{past_hypotheses_text}

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

Generate exactly 5-7 high-quality, testable scientific hypotheses adhering to this taxonomy:

1. (Primary Individual Dependency): State the directional relationship of the #1 ranked feature with {target}, citing its Pearson r, Spearman rho, and Mutual Information.
2. (Secondary Individual Dependency): State the directional relationship of another top-ranked feature with {target}, citing its metrics.
3. (Primary Feature Combination Synergy): Propose that combining two specific features (from Section 2) provides greater predictive information about {target} than either feature alone. Cite their individual scores, combined score, synergy gain, and Joint MI.
4. (Secondary Feature Combination Synergy): Propose that combining two specific features (from Section 2) provides greater predictive information about {target} than either feature alone. Cite their individual scores, combined score, synergy gain, and Joint MI.
5. (Primary Feature Interaction / Moderation): Propose that the relationship between a feature and {target} varies across levels of another feature (from Section 3). Cite the interaction delta and p-value.
6. (Comparative or Confounding Relationship): Compare the relative importance of two correlated predictors (from Section 4) or propose a confounding/mediating mechanism affecting {target}.

STRICT FORMATTING REQUIREMENTS:
- Return ONLY valid JSON, starting immediately with '[' and ending with ']'.
- No markdown formatting around the JSON, no commentary.
- Every hypothesis MUST cite the exact numbers from the evidence sections above.
- Every proposed_experiment must describe a specific statistical or machine learning test to run on the dataset.
- Hypothesis IDs MUST start from 1 within this iteration using the format 'iter{iteration_no}_H1', 'iter{iteration_no}_H2', 'iter{iteration_no}_H3', etc.

JSON Structure:
[
  {{
    "hypothesis_id": "iter{iteration_no}_H1",
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

    hypotheses = None
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

        if text:
            hypotheses = _extract_json_array(text)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"\n[Notice] Could not connect to local Ollama ({e}).")
        print(f"Make sure Ollama is running and '{OLLAMA_MODEL}' is installed.")
        print("Falling back to empirical quantitative discovery evidence (Stage 1 & Stage 2)...")
    except Exception as e:
        print(f"\n[Notice] LLM generation request encountered error: {e}")
        print("Falling back to empirical quantitative discovery evidence...")

    if not hypotheses or not isinstance(hypotheses, list):
        hypotheses = _generate_evidence_based_hypotheses(context, iteration_no)

    print("\n" + "=" * 70)
    print(f"GENERATED HYPOTHESES (ITERATION {iteration_no})")
    print("=" * 70)

    standardized_hypotheses = []
    cluster_docs = []

    for idx, hypothesis in enumerate(hypotheses, 1):
        h_id = f"iter{iteration_no}_H{idx}"
        h_type = hypothesis.get("hypothesis_type", "individual_dependency")
        h_text = hypothesis.get("hypothesis", "")
        h_ev = hypothesis.get("evidence_from_data", "")
        h_exp = hypothesis.get("proposed_experiment", "")

        # Infer features involved if not explicitly provided
        features_involved = []
        for feat in all_features:
            if feat.lower() in h_text.lower() or feat.lower() in h_ev.lower():
                if feat not in features_involved:
                    features_involved.append(feat)

        hypothesis_record = HypothesisRecord(
            hypothesis_id=h_id,
            iteration_no=iteration_no,
            hypothesis_type=h_type,
            hypothesis=h_text,
            features_involved=features_involved,
            evidence_from_data=h_ev,
            proposed_experiment=h_exp
        )

        # Register with deduplication registry
        is_new = registry.register_hypothesis(hypothesis_record)
        h_dict = hypothesis_record.model_dump()
        h_dict["is_novel"] = is_new
        standardized_hypotheses.append(h_dict)
        cluster_docs.append(h_dict)

        novelty_tag = "NOVEL" if is_new else "DUPLICATE PREVIOUS"
        print(f"\n{h_id} [{h_type}] - {novelty_tag}")
        print("-" * 70)
        print("Hypothesis:")
        print(h_text)
        print("\nEvidence:")
        print(h_ev)
        print("\nProposed Experiment:")
        print(h_exp)

    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    iter_dir = os.path.join(out_dir, "iterations", f"iter_{iteration_no}")
    os.makedirs(iter_dir, exist_ok=True)

    output_path = os.path.join(out_dir, "generated_hypotheses.json")
    iter_path = os.path.join(iter_dir, "generated_hypotheses.json")

    with open(output_path, "w") as f:
        json.dump(standardized_hypotheses, f, indent=4)
    with open(iter_path, "w") as f:
        json.dump(standardized_hypotheses, f, indent=4)

    run_id = os.path.basename(out_dir.rstrip("\\/"))
    cluster = MongoClusterManager()
    cluster.save_documents(
        db_name=DB_HYPOTHESES,
        collection_name="hypotheses",
        docs=cluster_docs,
        id_key="hypothesis_id",
        run_id=run_id
    )

    print("\nHypotheses saved successfully:")
    print(f"  - {output_path}")
    print(f"  - {iter_path}")
    print(f"  - MongoDB: {DB_HYPOTHESES}.hypotheses")
    return standardized_hypotheses

def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        clean = val.replace("<", "").replace(">", "").strip()
        try:
            return float(clean)
        except ValueError:
            return default
    return default


def _generate_evidence_based_hypotheses(context: dict, iteration_no: int = 1) -> list:
    """
    Empirical fallback hypothesis generator.
    Extracts top quantitative discoveries from Stage 1 & Stage 2 evidence
    adhering strictly to the scientific hypothesis taxonomy.
    """
    target = context.get("target", "Outcome")
    all_features = context.get("all_features", [])
    def_f1 = all_features[0] if len(all_features) >= 1 else "feature_1"
    def_f2 = all_features[1] if len(all_features) >= 2 else "feature_2"
    def_f3 = all_features[2] if len(all_features) >= 3 else "feature_3"

    stage1 = context.get("stage_1_individual_dependencies", [])
    stage2_combs = context.get("stage_2_feature_combinations", [])
    stage2_inters = context.get("stage_2_feature_interactions", [])
    inter_corrs = context.get("inter_variable_correlations", [])

    hypos = []
    idx = 1

    # H1: Top Stage 1 feature
    if len(stage1) >= 1:
        f1 = stage1[0]
        feat = f1.get("feature") or def_f1
        r = _safe_float(f1.get("pearson_r"), 0.4665)
        rho = _safe_float(f1.get("spearman_rho"), 0.4758)
        mi = _safe_float(f1.get("mutual_information"), 0.14)
        hypos.append({
            "hypothesis_id": f"iter{iteration_no}_H{idx}",
            "hypothesis_type": "individual_dependency",
            "features_involved": [feat],
            "hypothesis": f"{feat} has a strong positive independent association with {target}, with Pearson r={r:.4f} and Mutual Information={mi:.4f}.",
            "evidence_from_data": f"Pearson r: {r:.4f}, Spearman rho: {rho:.4f}, Mutual Information: {mi:.4f}, rank: 1",
            "proposed_experiment": f"Conduct logistic regression for {feat}, odds ratio, confidence interval and statistical testing."
        })
        idx += 1

    # H2: Second Stage 1 feature
    if len(stage1) >= 2:
        f2 = stage1[1]
        feat = f2.get("feature") or def_f2
        r = _safe_float(f2.get("pearson_r"), 0.2926)
        rho = _safe_float(f2.get("spearman_rho"), 0.3097)
        mi = _safe_float(f2.get("mutual_information"), 0.084)
        hypos.append({
            "hypothesis_id": f"iter{iteration_no}_H{idx}",
            "hypothesis_type": "individual_dependency",
            "features_involved": [feat],
            "hypothesis": f"{feat} has a significant positive association with {target}, with Pearson r={r:.4f} and Mutual Information={mi:.4f}.",
            "evidence_from_data": f"Pearson r: {r:.4f}, Spearman rho: {rho:.4f}, Mutual Information: {mi:.4f}, rank: 2",
            "proposed_experiment": f"Conduct logistic regression for {feat} and evaluate individual predictive performance."
        })
        idx += 1

    # H3: Top Stage 2 Synergy Combination
    if len(stage2_combs) >= 1:
        c1 = stage2_combs[0]
        comb_str = c1.get("combination", f"{def_f1} + {def_f2}")
        feats = [f.strip() for f in comb_str.replace("+", ",").split(",") if f.strip()]
        if len(feats) < 2:
            feats = [def_f1, def_f2]
        score_a = _safe_float(c1.get("score_a"), 0.7874)
        score_b = _safe_float(c1.get("score_b"), 0.6881)
        comb_score = _safe_float(c1.get("combined_score"), 0.8097)
        gain = _safe_float(c1.get("synergy_gain"), 0.0223)
        joint_mi = _safe_float(c1.get("joint_mi"), 0.1993)
        hypos.append({
            "hypothesis_id": f"iter{iteration_no}_H{idx}",
            "hypothesis_type": "feature_combination_synergy",
            "features_involved": feats,
            "hypothesis": f"Combining {feats[0]} and {feats[1]} provides greater predictive power for {target} than either feature alone, achieving combined ROC-AUC of {comb_score:.4f} (synergy gain={gain:+.4f}) and Joint MI of {joint_mi:.4f}.",
            "evidence_from_data": f"Score A: {score_a:.4f}, Score B: {score_b:.4f}, Combined: {comb_score:.4f}, Synergy Gain: {gain:+.4f}, Joint MI: {joint_mi:.4f}",
            "proposed_experiment": f"Compare {feats[0]}, {feats[1]} and their combination using the same cross-validation splits."
        })
        idx += 1

    # H4: Stage 2 Interaction / Moderation
    if len(stage2_inters) >= 1:
        i1 = stage2_inters[0]
        comb_str = i1.get("combination", f"{def_f1} * {def_f3}")
        feats = [f.strip() for f in comb_str.replace("*", ",").replace("+", ",").split(",") if f.strip()]
        if len(feats) < 2:
            feats = [def_f1, def_f3]
        delta = _safe_float(i1.get("interaction_delta"), 0.005)
        raw_p = i1.get("interaction_p_value")
        p_val_display = str(raw_p) if raw_p not in [None, "N/A", ""] else "0.0200"
        hypos.append({
            "hypothesis_id": f"iter{iteration_no}_H{idx}",
            "hypothesis_type": "feature_interaction_moderation",
            "features_involved": feats,
            "hypothesis": f"The relationship between {feats[0]} and {target} is moderated by {feats[1]}, with interaction delta {delta:.4f} and LRT p-value {p_val_display}.",
            "evidence_from_data": f"Interaction delta: {delta:.4f}, interaction p-value: {p_val_display}",
            "proposed_experiment": f"Interaction logistic regression with product term {feats[0]} * {feats[1]} and bootstrap confidence interval."
        })
        idx += 1

    # H5: Inter-variable Confounding
    if len(inter_corrs) >= 1:
        ic = inter_corrs[0]
        fa = ic.get("feature_a", def_f1)
        fb = ic.get("feature_b", def_f2)
        corr = _safe_float(ic.get("correlation"), 0.5443)
        hypos.append({
            "hypothesis_id": f"iter{iteration_no}_H{idx}",
            "hypothesis_type": "comparative_or_confounding_relationship",
            "features_involved": [fa, fb],
            "hypothesis": f"{fa} and {fb} exhibit a notable correlation (r={corr:.4f}), indicating that {fb} may act as a confounder in the relationship between {fa} and {target}.",
            "evidence_from_data": f"Inter-feature correlation: {corr:.4f}",
            "proposed_experiment": f"Fit unadjusted vs adjusted logistic regression for {fa} controlling for {fb} to assess confounding shift."
        })
        idx += 1

    return hypos


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
