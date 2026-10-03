import os
import json
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def create_llm_context(df, target, patterns):
    evidence = []
    for pattern in patterns:
        if pattern.get("type") == "correlation":
            evidence.append({
                "type": "statistical_relationship",
                "feature": pattern.get("feature"),
                "target": pattern.get("target"),
                "correlation": pattern.get("correlation"),
                "p_value": pattern.get("p_value"),
                "direction": pattern.get("direction")
            })
        elif pattern.get("type") == "ml_feature_importance":
            evidence.append({
                "type": "feature_importance",
                "feature": pattern.get("feature"),
                "importance": pattern.get("importance")
            })
        elif pattern.get("type") == "group_difference":
            evidence.append({
                "type": "group_difference",
                "feature": pattern.get("feature"),
                "target": pattern.get("target"),
                "group_means": pattern.get("group_means"),
                "p_value": pattern.get("p_value")
            })
        elif pattern.get("type") == "categorical_association":
            evidence.append({
                "type": "categorical_association",
                "feature": pattern.get("feature"),
                "target": pattern.get("target"),
                "chi_square": pattern.get("chi_square"),
                "p_value": pattern.get("p_value")
            })

    evidence = evidence[:8]

    context = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "target": str(target),
        "discovered_evidence": evidence
    }

    with open(os.path.join(OUTPUT_DIR, "llm_context.json"), "w") as f:
        json.dump(context, f, indent=4)

    return context
