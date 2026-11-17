import os
import json
import logging
from typing import List, Dict, Any, Optional
from .config import HYPOTHESIS_REGISTRY_PATH, DB_HYPOTHESES
from .schemas import HypothesisRecord
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)

class HypothesisRegistry:
    """
    Maintains a deduplicated index of all hypotheses across iterations.
    Prevents generating redundant hypotheses in subsequent iteration loops.
    """
    def __init__(
        self,
        registry_path: str = HYPOTHESIS_REGISTRY_PATH,
        run_id: Optional[str] = None
    ):
        self.registry_path = registry_path
        norm_dir = os.path.basename(os.path.dirname(os.path.abspath(registry_path)))
        self.run_id = run_id or (norm_dir if norm_dir != "ai_scientist_output" else None)
        self._hypotheses: List[HypothesisRecord] = []
        self._fingerprints: set[str] = set()
        self._load_registry()

    def _load_registry(self) -> None:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        rec = HypothesisRecord(**item)
                        self._hypotheses.append(rec)
                        if rec.fingerprint:
                            self._fingerprints.add(rec.fingerprint)
            except Exception as e:
                logger.warning(f"Failed to load registry from {self.registry_path}: {e}")

    def compute_fingerprint(self, hypothesis: HypothesisRecord) -> str:
        if hypothesis.fingerprint:
            return hypothesis.fingerprint
        sorted_feats = tuple(sorted(hypothesis.features_involved))
        return f"{sorted_feats}-{hypothesis.hypothesis_type}"

    def is_duplicate(self, hypothesis: HypothesisRecord) -> bool:
        fp = self.compute_fingerprint(hypothesis)
        return fp in self._fingerprints

    def register_hypothesis(self, hypothesis: HypothesisRecord) -> bool:
        """
        Registers hypothesis if novel. Returns True if registered, False if duplicate.
        """
        fp = self.compute_fingerprint(hypothesis)
        hypothesis.fingerprint = fp
        if fp in self._fingerprints:
            return False

        self._fingerprints.add(fp)
        self._hypotheses.append(hypothesis)
        self._save_registry()
        return True

    def _save_registry(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.registry_path)), exist_ok=True)
        docs = [h.model_dump() for h in self._hypotheses]
        try:
            with open(self.registry_path, "w") as f:
                json.dump(docs, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save registry to file: {e}")

        cluster = MongoClusterManager()
        cluster.save_documents(
            db_name=DB_HYPOTHESES,
            collection_name="hypotheses",
            docs=docs,
            id_key="hypothesis_id",
            run_id=self.run_id
        )

    def get_all_hypotheses(self) -> List[HypothesisRecord]:
        return list(self._hypotheses)

    def get_hypotheses_for_iteration(self, iteration_no: int) -> List[HypothesisRecord]:
        return [h for h in self._hypotheses if h.iteration_no == iteration_no]

    def get_past_hypotheses_prompt_summary(self) -> str:
        """
        Formats past hypotheses as bullet points to guide the LLM against duplication.
        """
        if not self._hypotheses:
            return "None (First iteration - generate primary hypotheses)."

        lines = []
        for h in self._hypotheses:
            feats = ", ".join(h.features_involved) if h.features_involved else "None"
            lines.append(f"- [{h.hypothesis_id} | Iteration {h.iteration_no}] Type: {h.hypothesis_type} | Features: ({feats}) -> {h.hypothesis}")
        return "\n".join(lines)
