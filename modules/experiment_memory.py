import os
import json
import logging
from typing import List, Dict, Any, Optional, Set, Union
from .config import (
    OUTPUT_DIR,
    DB_EXPERIMENTS,
    DB_RESULTS,
    DB_HYPOTHESES,
    DB_MEMORY,
    EXPERIMENT_SPECS_PATH,
    EXPERIMENT_RESULTS_PATH
)
from .schemas import (
    ExperimentSpecRecord,
    HypothesisRecord,
    StandardExperimentResult,
    NormalizedExperimentFingerprint
)
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)


class ExperimentMemory:
    """
    Batch 7: MongoDB Experiment Memory & Normalized Duplicate Prevention.
    Maintains a persistent record of all planned experiments, executed results,
    and normalized fingerprints across iterations to prevent duplicate executions (Batch 6.3).
    """
    def __init__(self, storage_path: Optional[str] = None, sync_mongo: bool = True):
        self.storage_path = storage_path or os.path.join(OUTPUT_DIR, "experiment_memory.json")
        self.sync_mongo = sync_mongo
        self.fingerprints: Set[str] = set()
        self.hypotheses: Dict[str, Dict[str, Any]] = {}
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self._load_memory()

    def _load_memory(self) -> None:
        """
        Loads cached history from local storage and queries MongoDB if available.
        """
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.fingerprints = set(data.get("fingerprints", []))
                    self.hypotheses = data.get("hypotheses", {})
                    self.experiments = data.get("experiments", {})
                    self.results = data.get("results", {})
            except Exception as e:
                logger.warning(f"Could not load local experiment memory: {e}")

        # Also populate from MongoDB if cluster is active and sync_mongo is enabled
        if self.sync_mongo:
            try:
                cluster = MongoClusterManager()
                if cluster.is_available():
                    fp_docs = cluster.query_documents(DB_MEMORY, "fingerprints")
                    for doc in fp_docs:
                        fp = doc.get("fingerprint")
                        if fp:
                            self.fingerprints.add(fp)
                    exp_docs = cluster.query_documents(DB_MEMORY, "experiments")
                    for doc in exp_docs:
                        spec_id = doc.get("experiment_id") or doc.get("spec_id")
                        if spec_id:
                            self.experiments[spec_id] = doc
                    res_docs = cluster.query_documents(DB_MEMORY, "results")
                    for doc in res_docs:
                        exp_id = doc.get("experiment_id")
                        if exp_id:
                            self.results[exp_id] = doc
            except Exception as e:
                logger.warning(f"Could not query MongoDB for experiment memory sync: {e}")

    def save_to_disk(self) -> None:
        """
        Persists memory state to disk.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            data = {
                "fingerprints": sorted(list(self.fingerprints)),
                "hypotheses": self.hypotheses,
                "experiments": self.experiments,
                "results": self.results
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist experiment memory to {self.storage_path}: {e}")

    @staticmethod
    def compute_normalized_fingerprint(
        features: List[str],
        model: str,
        target: str = "Outcome",
        metric: str = "ROC_AUC"
    ) -> str:
        """
        Computes a normalized configuration signature: sorted_features|model|target|metric.
        """
        clean_feats = [f.strip() for f in features if f.strip()]
        fp_obj = NormalizedExperimentFingerprint(
            features=clean_feats,
            model=model.strip(),
            target=target.strip(),
            metric=metric.strip()
        )
        return fp_obj.fingerprint or ""

    def is_duplicate_experiment(self, spec: Union[ExperimentSpecRecord, Dict[str, Any]]) -> bool:
        """
        Checks whether an experiment configuration has already been executed or scheduled.
        """
        if isinstance(spec, ExperimentSpecRecord):
            feats = spec.features
            model = spec.model_type
            target = spec.target
        else:
            feats = spec.get("features", [])
            model = spec.get("model_type") or spec.get("model", "")
            target = spec.get("target", "Outcome")

        fp = self.compute_normalized_fingerprint(feats, model, target)
        if fp in self.fingerprints:
            return True

        # Check in MongoDB if cluster is active and sync_mongo is enabled
        if self.sync_mongo:
            try:
                cluster = MongoClusterManager()
                if cluster.is_available():
                    col = cluster.get_collection(DB_MEMORY, "fingerprints")
                    if col is not None:
                        match = col.find_one({"fingerprint": fp})
                        if match:
                            self.fingerprints.add(fp)
                            return True
            except Exception as e:
                logger.debug(f"MongoDB duplicate check error: {e}")

        return False

    def register_experiment_spec(self, spec: Union[ExperimentSpecRecord, Dict[str, Any]]) -> None:
        """
        Registers an experiment specification in memory and marks its normalized fingerprint.
        """
        if isinstance(spec, ExperimentSpecRecord):
            spec_dict = spec.model_dump()
            spec_id = spec.experiment_id
            feats = spec.features
            model = spec.model_type
            target = spec.target
        else:
            spec_dict = spec
            spec_id = spec.get("experiment_id") or spec.get("spec_id", "E")
            feats = spec.get("features", [])
            model = spec.get("model_type") or spec.get("model", "")
            target = spec.get("target", "Outcome")

        fp = self.compute_normalized_fingerprint(feats, model, target)
        self.fingerprints.add(fp)
        self.experiments[spec_id] = spec_dict
        self.save_to_disk()

        # MongoDB persistence if sync_mongo enabled
        if self.sync_mongo:
            try:
                cluster = MongoClusterManager()
                if cluster.is_available():
                    fp_col = cluster.get_collection(DB_MEMORY, "fingerprints")
                    if fp_col is not None:
                        fp_col.update_one({"fingerprint": fp}, {"$set": {"fingerprint": fp, "experiment_id": spec_id}}, upsert=True)
                    exp_col = cluster.get_collection(DB_MEMORY, "experiments")
                    if exp_col is not None:
                        exp_col.update_one({"experiment_id": spec_id}, {"$set": spec_dict}, upsert=True)
            except Exception as e:
                logger.warning(f"Error syncing experiment spec to MongoDB: {e}")

    def save_hypothesis(self, hypothesis: Union[HypothesisRecord, Dict[str, Any]]) -> None:
        """
        Records a hypothesis in experiment memory.
        """
        h_dict = hypothesis.model_dump() if isinstance(hypothesis, HypothesisRecord) else hypothesis
        h_id = h_dict.get("hypothesis_id", "H")
        self.hypotheses[h_id] = h_dict
        self.save_to_disk()

        try:
            cluster = MongoClusterManager()
            if cluster.is_available():
                col = cluster.get_collection(DB_MEMORY, "hypotheses")
                if col is not None:
                    col.update_one({"hypothesis_id": h_id}, {"$set": h_dict}, upsert=True)
        except Exception as e:
            logger.warning(f"Error syncing hypothesis to MongoDB: {e}")

    def save_experiment_result(self, result: Union[StandardExperimentResult, Dict[str, Any]]) -> None:
        """
        Records an experiment evaluation result into memory.
        """
        r_dict = result.model_dump() if isinstance(result, StandardExperimentResult) else result
        exp_id = r_dict.get("experiment_id", "E")
        self.results[exp_id] = r_dict
        self.save_to_disk()

        try:
            cluster = MongoClusterManager()
            if cluster.is_available():
                col = cluster.get_collection(DB_MEMORY, "results")
                if col is not None:
                    col.update_one({"experiment_id": exp_id}, {"$set": r_dict}, upsert=True)
        except Exception as e:
            logger.warning(f"Error syncing result to MongoDB: {e}")

    def get_all_experiment_results(self) -> List[Dict[str, Any]]:
        """
        Returns all stored experiment results.
        """
        return list(self.results.values())

    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """
        Finds the highest-scoring experiment result based on ROC-AUC or metrics.
        """
        best_item = None
        best_score = -1.0

        for exp_id, res in self.results.items():
            results_data = res.get("results", {})
            score = (
                results_data.get("combined_auc") or
                results_data.get("roc_auc") or
                results_data.get("mean_cv_auc") or
                results_data.get("mean_auc") or
                0.0
            )
            if score > best_score:
                best_score = score
                best_item = {
                    "experiment_id": exp_id,
                    "hypothesis_id": res.get("hypothesis_id", ""),
                    "score": float(best_score),
                    "features": res.get("configuration", {}).get("features", []),
                    "model": res.get("configuration", {}).get("model", ""),
                    "raw_result": res
                }
        return best_item

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Returns aggregate statistics of scientific history.
        """
        supported = 0
        refuted = 0
        inconclusive = 0

        for res in self.results.values():
            verdict = res.get("evaluation", {}).get("verdict", "").lower()
            if "supported" in verdict and "not" not in verdict:
                supported += 1
            elif "not_supported" in verdict or "refuted" in verdict:
                refuted += 1
            else:
                inconclusive += 1

        best = self.get_best_result()
        return {
            "total_experiments": len(self.results),
            "supported_count": supported,
            "refuted_count": refuted,
            "inconclusive_count": inconclusive,
            "best_auc": best["score"] if best else 0.0,
            "best_features": best["features"] if best else []
        }
