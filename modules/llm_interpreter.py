import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def _safe_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

from .config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    OUTPUT_DIR,
    DB_REFINEMENTS,
    LLM_INTERPRETATIONS_PATH,
    REFINED_HYPOTHESES_PATH
)
from .schemas import (
    StandardExperimentResult,
    HypothesisRecord,
    LLMInterpretationRecord,
    RefinedHypothesisRecord
)
from .hypothesis_registry import HypothesisRegistry
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)


def build_interpretation_prompt(
    result: StandardExperimentResult,
    hypothesis: Optional[HypothesisRecord] = None
) -> str:
    """
    Builds a role-driven, evidence-based interpretation prompt for the local LLM (Batch 5.2).
    """
    if hypothesis:
        if isinstance(hypothesis, dict):
            hypo_text = hypothesis.get("hypothesis") or hypothesis.get("statement") or f"Hypothesis linked to {result.experiment_id}"
        else:
            hypo_text = getattr(hypothesis, "hypothesis", None) or getattr(hypothesis, "statement", None) or f"Hypothesis linked to {result.experiment_id}"
    else:
        hypo_text = f"Hypothesis linked to {result.experiment_id}"
    features_tested = ", ".join(result.configuration.get("features", []))
    model_name = result.configuration.get("model", "LogisticRegression")
    
    results_str = json.dumps(result.results, indent=2)
    stat_val_str = json.dumps(result.statistical_validation, indent=2)
    eval_str = json.dumps(result.evaluation, indent=2)

    prompt = f"""Role
You are the interpretation component of an autonomous AI Scientist.

Original Hypothesis ({result.hypothesis_id})
{hypo_text}

Experiment Configuration
- Experiment ID: {result.experiment_id}
- Experiment Type: {result.experiment_type}
- Model: {model_name}
- Features tested: {features_tested}
- Target: {result.configuration.get('target', 'Outcome')}

Experimental Results
{results_str}

Statistical Validation
{stat_val_str}

Evaluation Summary
{eval_str}

Instructions
1. Interpret the experimental findings.
2. Determine whether the hypothesis is supported, not supported, or inconclusive.
3. Identify the strongest observations.
4. Do not invent statistical significance or causal relationships.
5. Suggest one or more testable follow-up hypotheses.
6. Propose a suitable next experiment.

Return a structured JSON response containing:
- hypothesis_status: (must be one of 'supported_by_current_evidence', 'not_supported', 'inconclusive')
- interpretation: (clear concise interpretation of findings)
- observations: (list of specific factual observations citing numbers)
- limitations: (list of methodological, causal, or dataset limitations)
- next_hypothesis: (specific, testable follow-up hypothesis for the next research iteration)
- proposed_experiment: (object containing 'features', 'model', and 'metric')

STRICT FORMATTING REQUIREMENTS:
- Return ONLY valid JSON, starting immediately with '{{' and ending with '}}'.
- No markdown formatting around the JSON, no commentary.
"""
    return prompt


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Finds the first valid JSON object in text and returns the parsed dict.
    Robustly handles duplicated braces or markdown formatting.
    """
    clean_text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    starts = [i for i, ch in enumerate(clean_text) if ch == "{"]
    ends = [i for i, ch in enumerate(clean_text) if ch == "}"]

    for start in starts:
        for end in reversed(ends):
            if end > start:
                candidate = clean_text[start:end + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue

    return None


def interpret_single_experiment(
    result: StandardExperimentResult,
    hypothesis: Optional[HypothesisRecord] = None,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL
) -> LLMInterpretationRecord:
    """
    Queries local LLM via Ollama to generate an interpretation, observations,
    limitations, and next-hypothesis recommendation for a single experiment (Batch 5.1 & 5.2).
    Fails with ConnectionError if Ollama is unreachable.
    """
    prompt = build_interpretation_prompt(result, hypothesis)

    is_qwen = "qwen" in model.lower()
    if is_qwen:
        formatted_prompt = f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n{{\n"
        prefix = "{\n"
        use_raw = True
    else:
        formatted_prompt = prompt
        prefix = ""
        use_raw = False

    payload = {
        "model": model,
        "prompt": formatted_prompt,
        "raw": use_raw,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2500
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        res_data = response.json()
        resp_str = res_data.get("response", "").strip()
        if prefix and not resp_str.startswith("{"):
            raw_text = prefix + resp_str
        else:
            raw_text = resp_str
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise ConnectionError(f"Could not connect to Ollama at {url}: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama interpretation request failed: {e}")


    parsed = _extract_json_object(raw_text)
    if parsed is None:
        logger.warning(f"Failed to parse JSON object from LLM response for {result.experiment_id}: {raw_text[:200]}")
        parsed = {}

    status = parsed.get("hypothesis_status")
    if status not in ("supported_by_current_evidence", "not_supported", "inconclusive"):
        # Map fallback based on result evaluation verdict
        eval_verdict = result.evaluation.get("verdict", "inconclusive")
        status = eval_verdict if eval_verdict in ("supported_by_current_evidence", "not_supported", "inconclusive") else "inconclusive"

    interpretation = parsed.get("interpretation") or result.evaluation.get("summary", "")
    observations = parsed.get("observations") or [result.evaluation.get("summary", "")]
    limitations = parsed.get("limitations") or ["Requires external cohort validation."]
    next_hypo = parsed.get("next_hypothesis") or ""
    proposed_exp = parsed.get("proposed_experiment") or {}

    interp_id = f"INT-{result.experiment_id}"

    return LLMInterpretationRecord(
        interpretation_id=interp_id,
        experiment_id=result.experiment_id,
        hypothesis_id=result.hypothesis_id,
        iteration_no=result.iteration_no,
        hypothesis_status=status,
        interpretation=interpretation,
        observations=observations,
        limitations=limitations,
        next_hypothesis=next_hypo,
        proposed_experiment=proposed_exp
    )


def register_refined_hypotheses(
    interpretations: List[LLMInterpretationRecord],
    next_iteration_no: int = 2,
    registry: Optional[HypothesisRegistry] = None
) -> List[RefinedHypothesisRecord]:
    """
    Takes LLM interpretations, extracts proposed next hypotheses, formats them
    with iteration-prefixed IDs ('iter<N+1>_H1', etc.), and registers them with
    HypothesisRegistry for deduplication in subsequent iterations (Batch 5.3 & 5.4).
    """
    if registry is None:
        registry = HypothesisRegistry()

    refined_records: List[RefinedHypothesisRecord] = []
    idx = 1

    for interp in interpretations:
        next_text = interp.next_hypothesis.strip()
        if not next_text:
            continue

        proposed_exp = interp.proposed_experiment or {}
        features = proposed_exp.get("features", [])

        if not features:
            known_feats = ["Glucose", "BMI", "Age", "Pregnancies", "BloodPressure", "Insulin", "SkinThickness", "DiabetesPedigreeFunction"]
            for f in known_feats:
                if f.lower() in next_text.lower() and f not in features:
                    features.append(f)

        h_id = f"iter{next_iteration_no}_H{idx}"
        rationale = f"Derived from {interp.experiment_id} ({interp.hypothesis_id}) outcome: {interp.interpretation[:150]}"

        rec = RefinedHypothesisRecord(
            hypothesis_id=h_id,
            parent_hypothesis_id=interp.hypothesis_id,
            iteration_no=next_iteration_no,
            hypothesis=next_text,
            rationale=rationale,
            proposed_experiment=proposed_exp,
            features_involved=features
        )
        refined_records.append(rec)

        # Register with HypothesisRegistry
        hypo_for_reg = HypothesisRecord(
            hypothesis_id=h_id,
            iteration_no=next_iteration_no,
            hypothesis_type="refined_follow_up",
            hypothesis=next_text,
            features_involved=features,
            evidence_from_data=interp.interpretation,
            proposed_experiment=json.dumps(proposed_exp) if isinstance(proposed_exp, dict) else str(proposed_exp)
        )
        registry.register_hypothesis(hypo_for_reg)
        idx += 1

    return refined_records


def interpret_and_refine_all(
    eval_results: List[StandardExperimentResult],
    hypotheses: List[HypothesisRecord],
    iteration_no: int = 1,
    output_dir: Optional[str] = None,
    registry: Optional[HypothesisRegistry] = None,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL
) -> Tuple[List[LLMInterpretationRecord], List[RefinedHypothesisRecord]]:
    """
    Orchestrates Batch 5: Passes each evaluated experiment to the local LLM for
    interpretation, observational synthesis, limitations analysis, and next-hypothesis
    formulation. Registers refined hypotheses for iteration N+1, saves outputs locally,
    and syncs to MongoDB cluster 'ai_scientist_refinements'.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 8: BATCH 5 — LLM INTERPRETATION & HYPOTHESIS REFINEMENT (ITERATION {iteration_no})")
    print("=" * 70)

    # Index hypotheses by hypothesis_id (supporting both HypothesisRecord objects and dicts)
    hypo_map: Dict[str, Any] = {}
    for h in hypotheses:
        if isinstance(h, dict):
            hid = h.get("hypothesis_id")
            if hid:
                hypo_map[hid] = HypothesisRecord(
                    hypothesis_id=hid,
                    iteration_no=h.get("iteration_no", iteration_no),
                    hypothesis_type=h.get("hypothesis_type", "individual_dependency"),
                    hypothesis=h.get("hypothesis") or h.get("statement") or "",
                    features_involved=h.get("features_involved", []),
                    evidence_from_data=str(h.get("evidence_from_data") or h.get("evidence") or ""),
                    proposed_experiment=str(h.get("proposed_experiment") or "")
                )
        elif hasattr(h, "hypothesis_id"):
            hypo_map[h.hypothesis_id] = h

    interpretations: List[LLMInterpretationRecord] = []
    cluster_interp_docs: List[Dict[str, Any]] = []

    for r in eval_results:
        matching_h = hypo_map.get(r.hypothesis_id)
        print(f"\nPrompting LLM to interpret {r.experiment_id} ({r.hypothesis_id})...")
        interp = interpret_single_experiment(result=r, hypothesis=matching_h, model=model, url=url)
        interpretations.append(interp)
        cluster_interp_docs.append(interp.model_dump())

        status_str = interp.hypothesis_status.upper()
        _safe_print(f"  Status       : {status_str}")
        _safe_print(f"  Interpretation: {interp.interpretation}")
        if interp.observations:
            _safe_print("  Observations :")
            for obs in interp.observations:
                _safe_print(f"    - {obs}")
        if interp.limitations:
            _safe_print("  Limitations  :")
            for lim in interp.limitations:
                _safe_print(f"    - {lim}")
        if interp.next_hypothesis:
            _safe_print(f"  Next Hypo    : {interp.next_hypothesis}")

    # Register refined hypotheses for next iteration (iteration_no + 1)
    next_iter = iteration_no + 1
    refined_records = register_refined_hypotheses(
        interpretations=interpretations,
        next_iteration_no=next_iter,
        registry=registry
    )

    _safe_print("\n" + "-" * 70)
    _safe_print(f"REFINED CANDIDATE HYPOTHESES FOR ITERATION {next_iter}: Total = {len(refined_records)}")
    _safe_print("-" * 70)
    for ref in refined_records:
        _safe_print(f"\n{ref.hypothesis_id} (Parent: {ref.parent_hypothesis_id})")
        _safe_print(f"  Hypothesis: {ref.hypothesis}")
        _safe_print(f"  Features  : {ref.features_involved}")
        if ref.proposed_experiment:
            _safe_print(f"  Experiment: {ref.proposed_experiment}")

    # File persistence
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    iter_dir = os.path.join(out_dir, "iterations", f"iter_{iteration_no}")
    os.makedirs(iter_dir, exist_ok=True)

    interp_json = [i.model_dump() for i in interpretations]
    refined_json = [rf.model_dump() for rf in refined_records]

    # Save run-level
    with open(os.path.join(out_dir, "llm_interpretations.json"), "w", encoding="utf-8") as f:
        json.dump(interp_json, f, indent=2)
    with open(os.path.join(out_dir, "refined_next_hypotheses.json"), "w", encoding="utf-8") as f:
        json.dump(refined_json, f, indent=2)

    # Save iteration-level
    with open(os.path.join(iter_dir, "llm_interpretations.json"), "w", encoding="utf-8") as f:
        json.dump(interp_json, f, indent=2)
    with open(os.path.join(iter_dir, "refined_next_hypotheses.json"), "w", encoding="utf-8") as f:
        json.dump(refined_json, f, indent=2)

    # MongoDB cluster sync
    try:
        cluster = MongoClusterManager()
        if cluster.is_available():
            int_col = cluster.get_collection(DB_REFINEMENTS, "interpretations")
            if int_col is not None and cluster_interp_docs:
                for doc in cluster_interp_docs:
                    int_col.update_one(
                        {"interpretation_id": doc["interpretation_id"], "iteration_no": doc["iteration_no"]},
                        {"$set": doc},
                        upsert=True
                    )
            ref_col = cluster.get_collection(DB_REFINEMENTS, "refined_hypotheses")
            if ref_col is not None and refined_json:
                for doc in refined_json:
                    ref_col.update_one(
                        {"hypothesis_id": doc["hypothesis_id"], "iteration_no": doc["iteration_no"]},
                        {"$set": doc},
                        upsert=True
                    )
            print(f"Synced {len(cluster_interp_docs)} interpretations and {len(refined_json)} refined hypotheses to MongoDB: {DB_REFINEMENTS}")
    except Exception as e:
        logger.warning(f"Error persisting refinements to MongoDB cluster: {e}")

    return interpretations, refined_records



