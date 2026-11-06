import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import requests

from .config import OLLAMA_MODEL, OLLAMA_URL
from .schemas import HypothesisRecord, RefinedHypothesisRecord

logger = logging.getLogger(__name__)


def score_candidate_promise(
    hypothesis: Union[HypothesisRecord, RefinedHypothesisRecord, Dict[str, Any]],
    valid_columns: List[str],
    memory_stats: Optional[Dict[str, Any]] = None
) -> float:
    """
    Evaluates the promise, validity, and potential scientific value of a candidate hypothesis.
    Returns 0.0 if the hypothesis depends on non-existent or hallucinated columns.
    """
    if isinstance(hypothesis, (HypothesisRecord, RefinedHypothesisRecord)):
        feats = hypothesis.features_involved
        hypo_text = hypothesis.hypothesis
        exp = hypothesis.proposed_experiment
    else:
        feats = hypothesis.get("features_involved", [])
        hypo_text = hypothesis.get("hypothesis", hypothesis.get("statement", ""))
        exp = hypothesis.get("proposed_experiment", {})

    valid_set = set(c.strip().lower() for c in valid_columns)

    if not feats:
        return 0.0

    # Strict check: all features involved must be present in the dataset columns
    for f in feats:
        if f.strip().lower() not in valid_set:
            return 0.0

    score = 1.0

    # Multi-feature synergy bonus (testing 2-3 feature predictive models)
    if 2 <= len(feats) <= 3:
        score += 0.5

    # Memory bonus: reward building upon already discovered top performers
    if memory_stats:
        best_feats = set(f.strip().lower() for f in memory_stats.get("best_features", []))
        for f in feats:
            if f.strip().lower() in best_feats:
                score += 0.5

    # Actionable experiment specification bonus
    if isinstance(exp, dict) and exp.get("model"):
        score += 0.3
    elif isinstance(exp, str) and len(exp) > 10:
        score += 0.2

    return score


def rank_and_select_candidates(
    candidate_pool: List[Union[HypothesisRecord, RefinedHypothesisRecord, Dict[str, Any]]],
    valid_columns: List[str],
    memory_stats: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[Union[HypothesisRecord, RefinedHypothesisRecord, Dict[str, Any]]]:
    """
    Batch 6 Component A (LLM Experiment Selector & Prioritizer).
    Ranks candidate hypotheses by scientific promise, filtering out invalid features,
    and returning the top K candidate hypotheses.
    """
    scored: List[tuple[float, Any]] = []

    for item in candidate_pool:
        score = score_candidate_promise(item, valid_columns, memory_stats)
        if score > 0.0:
            scored.append((score, item))

    # Sort descending by promise score
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def prompt_llm_for_alternative_experiment(
    rejected_hypothesis: Any,
    rejection_reason: str,
    valid_columns: List[str],
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL
) -> Optional[RefinedHypothesisRecord]:
    """
    Prompts the local LLM to propose a valid, novel alternative experiment when a prior
    proposal was rejected by the Experiment Controller (e.g., duplicate or hallucinated columns).
    """
    if hasattr(rejected_hypothesis, "hypothesis"):
        hypo_str = rejected_hypothesis.hypothesis
        parent_id = getattr(rejected_hypothesis, "parent_hypothesis_id", "H1")
        iter_no = getattr(rejected_hypothesis, "iteration_no", 2)
    else:
        hypo_str = rejected_hypothesis.get("hypothesis", "")
        parent_id = rejected_hypothesis.get("parent_hypothesis_id", "H1")
        iter_no = rejected_hypothesis.get("iteration_no", 2)

    prompt = f"""Role: Autonomous AI Scientist Experiment Selector

The previous proposed experiment was REJECTED by the Experiment Controller.
Rejection Reason: {rejection_reason}
Rejected Hypothesis: {hypo_str}

STRICT CONSTRAINTS:
1. Available dataset features: {", ".join(valid_columns)}
2. DO NOT invent or use any features outside of the available dataset features listed above.
3. Propose a new, valid, and novel follow-up hypothesis.

Output ONLY a JSON object with this exact structure:
{{
  "hypothesis": "...",
  "rationale": "...",
  "features": ["Feature1", "Feature2"],
  "model": "sklearn.linear_model.LogisticRegression",
  "metric": "ROC_AUC"
}}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("response", "{}")
        parsed = json.loads(raw_text)

        feats = parsed.get("features", [])
        clean_feats = [f for f in feats if f in valid_columns]
        if not clean_feats:
            return None

        return RefinedHypothesisRecord(
            hypothesis_id=f"iter{iter_no}_H_alt",
            parent_hypothesis_id=parent_id,
            iteration_no=iter_no,
            hypothesis=parsed.get("hypothesis", "Alternative hypothesis"),
            rationale=parsed.get("rationale", f"Correction for {rejection_reason}"),
            proposed_experiment={
                "features": clean_feats,
                "model": parsed.get("model", "sklearn.linear_model.LogisticRegression"),
                "metric": parsed.get("metric", "ROC_AUC")
            },
            features_involved=clean_feats
        )
    except Exception as e:
        logger.info(f"LLM alternative query unavailable ({e}). Generating valid data-driven alternative configuration.")
        available = [c for c in valid_columns if c != "Outcome"]
        # Find combination of 2 valid features
        pair = available[:2] if len(available) >= 2 else available
        if not pair:
            return None
        return RefinedHypothesisRecord(
            hypothesis_id=f"iter{iter_no}_H_alt",
            parent_hypothesis_id=parent_id,
            iteration_no=iter_no,
            hypothesis=f"Alternative synergy hypothesis: multivariate model combining {', '.join(pair)}.",
            rationale=f"Data-driven alternative following controller feedback: {rejection_reason}",
            proposed_experiment={
                "features": pair,
                "model": "sklearn.linear_model.LogisticRegression",
                "metric": "ROC_AUC"
            },
            features_involved=pair
        )
