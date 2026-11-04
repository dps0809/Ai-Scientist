import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .config import OUTPUT_DIR, DB_PATTERNS
from .schemas import PatternRecord
from .mongo_cluster import MongoClusterManager

def save_patterns_with_ids(patterns: List[Dict[str, Any]], input_id: str, output_dir: Optional[str] = None, force_recompute: bool = False) -> List[Dict[str, Any]]:
    """
    Enriches discovered patterns with unique pattern_ids and input_id reference,
    writes them to discovered_patterns.json, and synchronizes to MongoDB (ai_scientist_patterns.patterns).
    """
    out_dir = output_dir or OUTPUT_DIR
    json_path = os.path.join(out_dir, "discovered_patterns.json")

    # If caller provided empty patterns list and cache exists, load cached patterns
    if os.path.exists(json_path) and not force_recompute and not patterns:
        try:
            with open(json_path, "r") as f:
                cached = json.load(f)
            if cached:
                print(f"[CACHE HIT] Loaded {len(cached)} discovered patterns from: {json_path}")
                return cached
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("8. DISCOVERED PATTERNS WITH UNIQUE IDENTIFIERS")
    print("=" * 70)

    enriched_patterns = []
    cluster_docs = []

    for i, pattern in enumerate(patterns, 1):
        pat_copy = dict(pattern)
        pat_id = f"PAT-{i:03d}"
        pat_copy["pattern_id"] = pat_id
        pat_copy["input_id"] = input_id
        pat_copy["timestamp"] = datetime.now(timezone.utc).isoformat()
        enriched_patterns.append(pat_copy)

        # Standardized schema conversion
        features = []
        if "feature" in pat_copy:
            feat_val = pat_copy["feature"]
            features = [f.strip() for f in feat_val.split("+")] if "+" in str(feat_val) else [str(feat_val)]
        elif "features" in pat_copy and isinstance(pat_copy["features"], list):
            features = pat_copy["features"]

        pat_record = PatternRecord(
            pattern_id=pat_id,
            input_id=input_id,
            pattern_type=str(pat_copy.get("type", "discovered_pattern")),
            features=features,
            metrics=pat_copy.get("metrics", {}),
            description=str(pat_copy.get("description", pat_copy.get("finding", ""))),
            timestamp=pat_copy["timestamp"]
        )
        cluster_docs.append(pat_record.model_dump())

    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "discovered_patterns.json")
    with open(path, "w") as f:
        json.dump(enriched_patterns, f, indent=4)

    cluster = MongoClusterManager()
    cluster.save_documents(
        db_name=DB_PATTERNS,
        collection_name="patterns",
        docs=cluster_docs,
        id_key="pattern_id"
    )

    print(f"\nTotal discovered patterns: {len(enriched_patterns)} (Linked to Input ID: {input_id})")
    for i, pattern in enumerate(enriched_patterns, 1):
        print(f"\nPattern {pattern['pattern_id']}:")
        print(json.dumps(pattern, indent=2))

    return enriched_patterns

def save_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return save_patterns_with_ids(patterns, input_id="INP-LEGACY")
