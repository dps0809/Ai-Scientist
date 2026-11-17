import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from .config import OUTPUT_DIR, VALIDATED_HYPOTHESES_PATH, DB_VALIDATION
from .schemas import ValidationStatus, ValidationResultRecord, ValidationReportRecord, HypothesisRecord
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {
    "individual_dependency",
    "feature_combination_synergy",
    "feature_interaction_moderation",
    "comparative_or_confounding_relationship",
    "comparative_confounding_relationship",
    "confounding",
    "mediation"
}

def extract_features_from_text(text: str, df_columns: List[str]) -> List[str]:
    """
    Extracts column names mentioned in hypothesis text or evidence.
    """
    found = []
    text_lower = text.lower()
    for col in df_columns:
        pattern = r"\b" + re.escape(col.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(col)
    return found

def validate_hypotheses(
    hypotheses: List[Dict[str, Any]],
    df: pd.DataFrame,
    target: str = "Outcome",
    stage1_data: Optional[Dict[str, Any]] = None,
    stage2_data: Optional[Dict[str, Any]] = None,
    iteration_no: int = 1,
    output_dir: Optional[str] = None
) -> ValidationReportRecord:
    """
    Validates all LLM-generated hypotheses without omission.
    Performs schema validation, column existence checks, numeric type checks,
    binary target verification, and evidence discrepancy reconciliation.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 4: BATCH 1 — HYPOTHESIS VALIDATION & RECONCILIATION (ITER {iteration_no})")
    print("=" * 70)

    results: List[ValidationResultRecord] = []
    target_is_valid = target in df.columns
    target_is_binary = target_is_valid and set(df[target].dropna().unique()).issubset({0, 1})

    for idx, hyp in enumerate(hypotheses, 1):
        h_id = hyp.get("hypothesis_id", f"H{idx}")
        h_type = hyp.get("hypothesis_type", "individual_dependency")
        h_text = hyp.get("hypothesis", "")
        h_ev = hyp.get("evidence_from_data", "")
        h_exp = hyp.get("proposed_experiment", "")

        status = ValidationStatus.VALIDATED
        reasons: List[str] = []
        reconciled_evidence: Dict[str, Any] = {}

        # 1. Target checks
        if not target_is_valid:
            status = ValidationStatus.REJECTED
            reasons.append(f"Target column '{target}' not found in dataset.")
        elif not target_is_binary:
            status = ValidationStatus.REJECTED
            reasons.append(f"Target column '{target}' must be binary with values in {{0, 1}}.")

        # 2. Hypothesis type support
        if h_type not in SUPPORTED_TYPES:
            status = ValidationStatus.REJECTED
            reasons.append(f"Unsupported hypothesis type '{h_type}'.")

        # 3. Feature extraction and validation
        features_involved = hyp.get("features_involved", [])
        if not features_involved:
            features_involved = extract_features_from_text(f"{h_text} {h_ev}", list(df.columns))

        # Check for hallucinated feature names in text if none matched
        if not features_involved:
            # Check if text contains capitalized words that look like features
            words = re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", h_text)
            potential = [w for w in words if w not in {"The", "Higher", "Lower", "Combining", "Outcome"}]
            if potential:
                status = ValidationStatus.REJECTED
                reasons.append(f"Feature(s) '{', '.join(potential)}' not found in dataset columns.")
            else:
                status = ValidationStatus.REJECTED
                reasons.append("No identifiable dataset features found in hypothesis.")
        else:
            for feat in features_involved:
                if feat not in df.columns:
                    status = ValidationStatus.REJECTED
                    reasons.append(f"Feature '{feat}' not found in dataset.")
                elif not pd.api.types.is_numeric_dtype(df[feat]):
                    status = ValidationStatus.REJECTED
                    reasons.append(f"Feature '{feat}' must be numeric.")

        # 4. Evidence consistency & discrepancy reconciliation
        # Specific check: H4 interaction delta & p-value conflicting claims
        if h_type == "feature_interaction_moderation" or "interaction" in h_text.lower():
            text_delta_match = re.search(r"interaction\s+delta\s*=\s*(-?[0-9.]+)", h_text, re.IGNORECASE)
            ev_delta_match = re.search(r"interaction_delta\s*=\s*(-?[0-9.]+)", h_ev, re.IGNORECASE)

            # Ground truth from stage2_data if available
            ground_truth_delta = None
            ground_truth_p = None
            if stage2_data and "combinations" in stage2_data:
                for comb in stage2_data["combinations"]:
                    c_feats = comb.get("combination", [])
                    if set(c_feats).issubset(set(features_involved)):
                        ground_truth_delta = comb.get("interaction_delta")
                        ground_truth_p = comb.get("interaction_p_value")
                        break

            if ev_delta_match and not ground_truth_delta:
                try:
                    ground_truth_delta = float(ev_delta_match.group(1))
                except ValueError:
                    pass

            ev_p_match = re.search(r"interaction_p_value\s*=\s*([0-9.]+)", h_ev, re.IGNORECASE)
            if ev_p_match and not ground_truth_p:
                try:
                    ground_truth_p = float(ev_p_match.group(1))
                except ValueError:
                    pass

            if text_delta_match and ground_truth_delta is not None:
                claimed_delta = float(text_delta_match.group(1))
                if abs(claimed_delta - ground_truth_delta) > 1e-4:
                    status = ValidationStatus.NEEDS_REVIEW
                    reconciled_evidence["interaction_delta"] = ground_truth_delta
                    if ground_truth_p is not None:
                        reconciled_evidence["interaction_p_value"] = ground_truth_p
                    reasons.append(
                        f"Discrepancy detected: Hypothesis claims interaction_delta={claimed_delta}, "
                        f"while calculated evidence shows interaction_delta={ground_truth_delta}. "
                        f"Reconciled to calculated evidence."
                    )

        # Comparative / Confounding check (H5):
        if (
            h_type in ("comparative_or_confounding_relationship", "comparative_confounding_relationship", "confounding")
            or "confound" in h_type
            or "comparative" in h_type
            or "mediation" in h_exp.lower()
        ):
            h_type = "comparative_or_confounding_relationship"
            if status != ValidationStatus.REJECTED:
                status = ValidationStatus.NEEDS_REVIEW
                reasons.append("Hypothesis suggests confounding/mediation; routed to adjusted regression verification.")
                corr_match = re.search(r"r\s*=\s*([0-9.]+)", h_text)
                if corr_match:
                    reconciled_evidence["correlation"] = float(corr_match.group(1))

        # Check missing values
        if features_involved and all(f in df.columns for f in features_involved):
            missing_counts = {f: int(df[f].isna().sum()) for f in features_involved}
            if any(v > 0 for v in missing_counts.values()):
                reconciled_evidence["missing_values"] = missing_counts

        val_id = f"VAL-{h_id}" if "iter" in h_id else f"VAL-{h_id}-iter{iteration_no}"
        result_record = ValidationResultRecord(
            validation_id=val_id,
            hypothesis_id=h_id,
            iteration_no=iteration_no,
            hypothesis_type=h_type,
            status=status,
            features_involved=features_involved,
            reconciled_evidence=reconciled_evidence,
            reasons=reasons
        )
        results.append(result_record)

        print(f"\n{h_id}: [{status.value}]")
        print(f"  Features Involved: {', '.join(features_involved)}")
        if reasons:
            print(f"  Notes/Reasons: {'; '.join(reasons)}")
        if reconciled_evidence:
            print(f"  Reconciled Evidence: {reconciled_evidence}")

    validated_cnt = sum(1 for r in results if r.status == ValidationStatus.VALIDATED)
    needs_review_cnt = sum(1 for r in results if r.status == ValidationStatus.NEEDS_REVIEW)
    rejected_cnt = sum(1 for r in results if r.status == ValidationStatus.REJECTED)

    report_id = f"REP-iter{iteration_no}"
    report = ValidationReportRecord(
        report_id=report_id,
        iteration_no=iteration_no,
        total=len(results),
        validated_count=validated_cnt,
        needs_review_count=needs_review_cnt,
        rejected_count=rejected_cnt,
        results=results
    )

    # Persist report to JSON
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    iter_dir = os.path.join(out_dir, "iterations", f"iter_{iteration_no}")
    os.makedirs(iter_dir, exist_ok=True)

    report_dump = report.model_dump()
    with open(os.path.join(out_dir, "validated_hypotheses.json"), "w") as f:
        json.dump(report_dump, f, indent=4)
    with open(os.path.join(iter_dir, "validated_hypotheses.json"), "w") as f:
        json.dump(report_dump, f, indent=4)

    # Persist to MongoDB cluster scoped by run_id
    run_id = os.path.basename(out_dir.rstrip("\\/"))
    cluster = MongoClusterManager()
    cluster.save_document(
        db_name=DB_VALIDATION,
        collection_name="reports",
        doc=report_dump,
        id_key="report_id",
        run_id=run_id
    )
    cluster.save_documents(
        db_name=DB_VALIDATION,
        collection_name="validations",
        docs=[r.model_dump() for r in results],
        id_key="validation_id",
        run_id=run_id
    )

    print("\n" + "-" * 70)
    print(f"VALIDATION SUMMARY: Total={report.total} | Validated={validated_cnt} | Needs Review={needs_review_cnt} | Rejected={rejected_cnt}")
    print(f"Saved report to: {VALIDATED_HYPOTHESES_PATH}")
    print(f"Saved to MongoDB: {DB_VALIDATION}.validations")
    print("-" * 70)

    return report
