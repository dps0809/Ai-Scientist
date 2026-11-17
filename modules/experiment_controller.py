import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd

from .config import (
    OUTPUT_DIR,
    DB_MEMORY,
    AUTONOMOUS_STATE_PATH,
    FINAL_REPORT_MD_PATH,
    FINAL_REPORT_JSON_PATH,
    MAX_AUTONOMOUS_EXPERIMENTS,
    MAX_AUTONOMOUS_ITERATIONS,
    MAX_RUNTIME_MINUTES
)
from .schemas import (
    StoppingConditionConfig,
    AutonomousStateRecord,
    ExperimentSpecRecord,
    StandardExperimentResult,
    HypothesisRecord,
    RefinedHypothesisRecord,
    FinalScientificReportRecord
)
from .experiment_memory import ExperimentMemory
from .experiment_selector import rank_and_select_candidates, prompt_llm_for_alternative_experiment
from .hypothesis_validator import validate_hypotheses
from .experiment_planner import plan_experiments
from .experiment_executor import execute_all_experiments
from .result_evaluator import evaluate_all_results
from .llm_interpreter import interpret_and_refine_all
from .hypothesis_registry import HypothesisRegistry
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)


class ExperimentController:
    """
    Batch 6 Component B: The central authority that validates proposed plans,
    enforces normalized duplicate prevention, monitors stopping criteria,
    tracks research progress, and compiles final scientific reports.
    """
    def __init__(
        self,
        config: Optional[StoppingConditionConfig] = None,
        memory: Optional[ExperimentMemory] = None,
        df: Optional[pd.DataFrame] = None,
        target: str = "Outcome",
        output_dir: Optional[str] = None
    ):
        self.config = config or StoppingConditionConfig()
        self.output_dir = output_dir or OUTPUT_DIR
        self.memory = memory or ExperimentMemory(
            storage_path=os.path.join(self.output_dir, "experiment_memory.json")
        )
        self.df = df
        self.target = target
        self.start_time = datetime.now(timezone.utc)
        folder_name = os.path.basename(self.output_dir.rstrip("\\/"))
        run_id = folder_name if folder_name and folder_name != "ai_scientist_output" else f"RUN-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = AutonomousStateRecord(
            run_id=run_id,
            experiments_completed=0,
            iterations_completed=0,
            consecutive_no_improvement=0,
            best_score=0.0,
            best_accuracy=0.0,
            status="INITIALIZED",
            start_time=self.start_time.isoformat()
        )
        self._persist_state()

    def validate_experiment_spec(self, spec: ExperimentSpecRecord) -> Tuple[bool, str]:
        """
        Validates an experiment specification against dataset constraints and models.
        """
        if self.df is not None:
            # Check target column
            if self.target not in self.df.columns:
                return False, f"Target column '{self.target}' not found in dataset."

            # Check binary target
            target_vals = set(self.df[self.target].dropna().unique())
            if not target_vals.issubset({0, 1}):
                return False, f"Target column '{self.target}' is not binary {{0, 1}}."

            # Check features exist in dataset
            df_cols = set(self.df.columns)
            missing = [f for f in spec.features if f not in df_cols]
            if missing:
                return False, f"Feature(s) {missing} not found in dataset columns."

        # Check supported model
        supported_models = {
            "statsmodels.Logit",
            "sklearn.linear_model.LogisticRegression",
            "LogisticRegression",
            "statsmodels.OLS",
            "sklearn.linear_model.RidgeCV"
        }
        if spec.model_type not in supported_models and not any(m in spec.model_type for m in supported_models):
            return False, f"Unsupported model type '{spec.model_type}'."

        return True, "Valid"

    def should_stop(self) -> Tuple[bool, str]:
        """
        Checks whether any of the defined stopping conditions have been met (Batch 6.4).
        """
        # 1. Maximum experiments limit
        if self.state.experiments_completed >= self.config.max_experiments:
            return True, f"Maximum experiments limit reached ({self.state.experiments_completed}/{self.config.max_experiments})."

        # 2. Maximum iterations limit
        if self.state.iterations_completed >= self.config.max_iterations:
            return True, f"Maximum iterations limit reached ({self.state.iterations_completed}/{self.config.max_iterations})."

        # 3. Maximum runtime limit
        elapsed_min = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 60.0
        if elapsed_min >= self.config.max_runtime_minutes:
            return True, f"Maximum runtime budget reached ({elapsed_min:.1f}/{self.config.max_runtime_minutes} minutes)."

        # 4. Target performance metric achieved
        if self.state.best_score >= self.config.target_roc_auc:
            return True, f"Target ROC-AUC achieved ({self.state.best_score:.4f} >= {self.config.target_roc_auc:.4f})."

        # 5. Consecutive rounds with no meaningful improvement
        if self.state.consecutive_no_improvement >= self.config.no_improvement_tolerance:
            return True, f"No improvement tolerance reached ({self.state.consecutive_no_improvement} consecutive experiments without >= {self.config.min_meaningful_gain:.4f} gain)."

        return False, "Continue"

    def update_progress(self, result: StandardExperimentResult) -> None:
        """
        Updates the internal state, checks gain significance, and saves state to disk/MongoDB.
        Tracks primary ROC-AUC metric and secondary Accuracy metric.
        """
        self.state.experiments_completed += 1
        results_data = result.results
        score = (
            results_data.get("combined_auc") or
            results_data.get("roc_auc") or
            results_data.get("mean_cv_auc") or
            results_data.get("mean_auc") or
            0.0
        )
        accuracy = (
            results_data.get("combined_accuracy") or
            results_data.get("accuracy") or
            results_data.get("mean_accuracy") or
            0.0
        )
        best_model = (
            results_data.get("best_model") or
            result.configuration.get("model", "")
        )

        # Inferential statistical tests (moderation LRT, confounding shift) do not yield predictive AUC
        # and must not be penalized as failed predictive gains (keeping stats and prediction separate)
        is_inferential_test = result.experiment_type in (
            "feature_interaction_moderation",
            "comparative_or_confounding_relationship"
        )

        gain = score - self.state.best_score

        if not is_inferential_test:
            if gain >= self.config.min_meaningful_gain:
                self.state.best_score = score
                self.state.best_accuracy = accuracy
                self.state.best_features = result.configuration.get("features", [])
                self.state.best_model = best_model
                self.state.consecutive_no_improvement = 0
            else:
                self.state.consecutive_no_improvement += 1

        self.state.history.append({
            "experiment_id": result.experiment_id,
            "hypothesis_id": result.hypothesis_id,
            "score": score,
            "accuracy": accuracy,
            "features": result.configuration.get("features", []),
            "model": best_model,
            "verdict": result.evaluation.get("verdict", "")
        })

        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self._persist_state()

    def _persist_state(self) -> None:
        """
        Saves current state to local JSON file and MongoDB scoped by run_id.
        """
        try:
            state_file = os.path.join(self.output_dir, "autonomous_state.json")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.model_dump(), f, indent=2)

            cluster = MongoClusterManager()
            if cluster.is_available():
                cluster.save_document(
                    db_name=DB_MEMORY,
                    collection_name="autonomous_state",
                    doc=self.state.model_dump(),
                    id_key="run_id",
                    run_id=self.state.run_id
                )
        except Exception as e:
            logger.warning(f"Failed to persist autonomous state: {e}")

    def generate_final_report(self) -> Tuple[str, Dict[str, Any]]:
        """
        Synthesizes a comprehensive, publication-grade final scientific research report
        documenting the complete autonomous lifecycle, confirmed discoveries, and lineage.
        """
        stats = self.memory.get_summary_stats()
        all_results = self.memory.get_all_experiment_results()
        run_exp_ids = {h["experiment_id"] for h in self.state.history}
        target_results = [r for r in all_results if r.get("experiment_id") in run_exp_ids] if run_exp_ids else all_results

        confirmed = []
        refuted = []
        multi_model_rows = []

        for r in target_results:
            verdict = r.get("evaluation", {}).get("verdict", "").lower()
            res_metrics = r.get("results", {})
            exp_info = {
                "experiment_id": r.get("experiment_id"),
                "hypothesis_id": r.get("hypothesis_id"),
                "features": r.get("configuration", {}).get("features", []),
                "summary": r.get("evaluation", {}).get("summary", ""),
                "metrics": res_metrics,
                "best_model": res_metrics.get("best_model", r.get("configuration", {}).get("model", ""))
            }
            if "supported" in verdict and "not" not in verdict:
                confirmed.append(exp_info)
            else:
                refuted.append(exp_info)

            # Extract model comparison if available
            m_comp = res_metrics.get("model_comparison", {})
            if m_comp:
                feats_str = ", ".join(r.get("configuration", {}).get("features", []))
                best_m = res_metrics.get("best_model", "")
                primary_score = res_metrics.get("combined_auc") or res_metrics.get("mean_auc") or 0.0
                sec_score = res_metrics.get("combined_accuracy") or res_metrics.get("mean_accuracy") or 0.0
                multi_model_rows.append({
                    "experiment_id": r.get("experiment_id"),
                    "features": feats_str,
                    "winning_model": best_m,
                    "primary_auc": primary_score,
                    "secondary_accuracy": sec_score,
                    "comparison": m_comp
                })

        report_data = {
            "run_id": self.state.run_id,
            "dataset": "diabetes.csv",
            "total_experiments": self.state.experiments_completed,
            "total_iterations": self.state.iterations_completed,
            "primary_metric": "ROC_AUC",
            "secondary_metric": "Accuracy",
            "best_roc_auc": self.state.best_score,
            "best_accuracy": self.state.best_accuracy,
            "best_features": self.state.best_features,
            "best_model": self.state.best_model,
            "stopping_reason": self.state.stopping_reason or "All tasks completed",
            "confirmed_count": len(confirmed),
            "refuted_count": len(refuted),
            "confirmed_hypotheses": confirmed,
            "refuted_hypotheses": refuted,
            "multi_model_tournaments": multi_model_rows
        }

        # Build Markdown Report
        md = f"""# Autonomous AI Scientist — Final Research Report

**Run ID:** `{self.state.run_id}`  
**Dataset:** `diabetes.csv`  
**Generated At:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Stopping Reason:** {report_data['stopping_reason']}  

---

## 1. Executive Summary
The Autonomous AI Scientist executed a closed-loop empirical research program:
- **Total Experiments Conducted:** {self.state.experiments_completed}
- **Autonomous Iterations Completed:** {self.state.iterations_completed}
- **Confirmed Discoveries (Supported):** {len(confirmed)}
- **Refuted Hypotheses:** {len(refuted)}
- **Top Predictive Performance (Primary Metric):** **ROC-AUC = {self.state.best_score:.4f}**
- **Secondary Metric Performance:** **Accuracy = {self.state.best_accuracy:.4f}**
- **Optimal Feature Set:** `{", ".join(self.state.best_features) if self.state.best_features else "N/A"}`
- **Optimal Model Architecture:** `{self.state.best_model or "N/A"}`

---

## 2. Confirmed Scientific Discoveries
"""
        for i, c in enumerate(confirmed, 1):
            feats_str = ", ".join(c.get("features", []))
            md += f"### Discovery {i}: {c.get('experiment_id')} ({c.get('hypothesis_id')})\n"
            md += f"- **Features Tested:** `{feats_str}`\n"
            md += f"- **Evaluation Summary:** {c.get('summary')}\n"
            metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in c.get('metrics', {}).items() if not isinstance(v, (list, dict)))
            md += f"- **Key Metrics:** {metrics_str}\n\n"

        if multi_model_rows:
            md += "## 3. Multi-Model Architecture Tournament (Logistic Regression vs Random Forest vs XGBoost)\n"
            md += "Candidate predictive models were simultaneously evaluated across identical 5-fold Stratified Cross-Validation splits:\n\n"
            md += "| Experiment | Features | Winning Model | Primary ROC-AUC | Secondary Accuracy | Model Breakdown (AUC / Accuracy) |\n"
            md += "|---|---|---|---|---|---|\n"
            for row in multi_model_rows:
                breakdowns = []
                for m_name, m_vals in row["comparison"].items():
                    m_auc = m_vals.get("combined_mean_auc") or m_vals.get("mean_auc") or 0.0
                    m_acc = m_vals.get("combined_mean_accuracy") or m_vals.get("mean_accuracy") or 0.0
                    breakdowns.append(f"**{m_name}**: {m_auc:.3f}/{m_acc:.3f}")
                bd_str = " <br> ".join(breakdowns)
                md += f"| `{row['experiment_id']}` | `{row['features']}` | **{row['winning_model']}** | **{row['primary_auc']:.4f}** | {row['secondary_accuracy']:.4f} | {bd_str} |\n"
            md += "\n"

        md += "## 4. Refuted Hypotheses & Negative Results\n"
        for i, r in enumerate(refuted, 1):
            feats_str = ", ".join(r.get("features", []))
            md += f"### Negative Result {i}: {r.get('experiment_id')} ({r.get('hypothesis_id')})\n"
            md += f"- **Features Tested:** `{feats_str}`\n"
            md += f"- **Evaluation Summary:** {r.get('summary')}\n\n"

        md += """## 5. Limitations & Methodological Integrity
1. **Predictive Synergy vs Statistical Interaction:** Model gains were evaluated under cross-validation separately from multiplicative product terms to prevent confounding predictive power with interaction effects.
2. **Multiple Model Comparison:** Logistic Regression, Random Forest, and XGBoost were tested under identical cross-validation folds to determine whether linear separation or non-linear tree-based boundaries best capture the underlying biological risk.
3. **Primary vs Secondary Metrics:** ROC-AUC acts as the primary evaluation criterion due to class imbalance resilience; Accuracy is reported alongside as an auxiliary operational metric.
4. **Multiple Testing & Integrity:** Candidate hypotheses underwent permutation testing and within-fold median imputation to strictly avoid data leakage.

---
*Report generated autonomously by AI Scientist Engine.*
"""

        # Save files
        report_md_path = os.path.join(self.output_dir, "final_scientific_report.md")
        report_json_path = os.path.join(self.output_dir, "final_scientific_report.json")

        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(md)

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        try:
            cluster = MongoClusterManager()
            if cluster.is_available():
                cluster.save_document(
                    db_name=DB_MEMORY,
                    collection_name="final_reports",
                    doc=report_data,
                    id_key="run_id",
                    run_id=self.state.run_id
                )
        except Exception as e:
            logger.warning(f"Error saving final report to MongoDB: {e}")

        return md, report_data


def run_autonomous_scientist(
    initial_hypotheses: List[Dict[str, Any]],
    df: pd.DataFrame,
    target: str = "Outcome",
    config: Optional[StoppingConditionConfig] = None,
    output_dir: Optional[str] = None,
    stage1_data: Optional[Dict[str, Any]] = None,
    stage2_data: Optional[Dict[str, Any]] = None,
    n_permutations: int = 1000,
    n_bootstrap: int = 500,
    sync_mongo: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Batch 6: Autonomous Research Loop.
    Executes the self-directed scientific discovery cycle:
    Evidence -> Hypothesis -> Validation -> Experiment -> Evaluation -> Interpretation -> Refinement -> Next Experiment
    """
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    cfg = config or StoppingConditionConfig()

    run_id = os.path.basename(out_dir.rstrip("\\/"))
    memory = ExperimentMemory(storage_path=os.path.join(out_dir, "experiment_memory.json"), run_id=run_id, sync_mongo=sync_mongo)
    controller = ExperimentController(config=cfg, memory=memory, df=df, target=target, output_dir=out_dir)
    registry = HypothesisRegistry(registry_path=os.path.join(out_dir, "hypothesis_registry.json"), run_id=run_id)

    print("\n" + "=" * 70)
    print("STARTING AUTONOMOUS SCIENTIFIC DISCOVERY LOOP")
    print(f"Max Experiments: {cfg.max_experiments} | Max Iterations: {cfg.max_iterations}")
    print(f"Target ROC-AUC : {cfg.target_roc_auc} | Tolerance: {cfg.no_improvement_tolerance}")
    print("=" * 70)

    # Initial candidate pool
    candidate_pool: List[Any] = list(initial_hypotheses)
    valid_columns = [c for c in df.columns if c != target]
    iteration_no = 1

    while True:
        should_stop, stop_reason = controller.should_stop()
        if should_stop:
            controller.state.status = "COMPLETED"
            controller.state.stopping_reason = stop_reason
            print(f"\n[AUTONOMOUS STOPPING CONDITION TRIGGERED]: {stop_reason}")
            break

        print(f"\n--- AUTONOMOUS ITERATION {iteration_no} (Candidate Pool: {len(candidate_pool)}) ---")

        # 1. Select top candidate hypotheses
        ranked_candidates = rank_and_select_candidates(
            candidate_pool=candidate_pool,
            valid_columns=valid_columns,
            memory_stats=memory.get_summary_stats(),
            top_k=min(5, len(candidate_pool))
        )

        if not ranked_candidates and candidate_pool:
            valid_set = set(c.strip().lower() for c in valid_columns)
            invalid_reasons = []
            for cand in candidate_pool:
                feats = getattr(cand, "features_involved", None) or (cand.get("features_involved", []) if isinstance(cand, dict) else [])
                hid = getattr(cand, "hypothesis_id", None) or (cand.get("hypothesis_id", "unknown") if isinstance(cand, dict) else "unknown")
                missing_feats = [f for f in feats if f.strip().lower() not in valid_set]
                if missing_feats:
                    reason = f"Proposed unmeasured feature(s) {missing_feats} not found in dataset columns."
                    print(f"[SELECTOR REJECTED] {hid}: {reason} (Valid dataset columns: {valid_columns})")
                    invalid_reasons.append((cand, reason))
                elif not feats:
                    reason = "No features specified in candidate hypothesis."
                    print(f"[SELECTOR REJECTED] {hid}: {reason}")
                    invalid_reasons.append((cand, reason))

            # Attempt recovery: prompt LLM for alternative experiment using valid columns
            recovered_candidates = []
            for cand, reason in invalid_reasons[:2]:
                alt = prompt_llm_for_alternative_experiment(
                    rejected_hypothesis=cand,
                    rejection_reason=reason,
                    valid_columns=valid_columns
                )
                if alt and getattr(alt, "features_involved", None):
                    recovered_candidates.append(alt)

            if recovered_candidates:
                print(f"[CONTROLLER]: Recovered {len(recovered_candidates)} valid candidate(s) grounded in dataset features.")
                candidate_pool = recovered_candidates
                ranked_candidates = rank_and_select_candidates(
                    candidate_pool=candidate_pool,
                    valid_columns=valid_columns,
                    memory_stats=memory.get_summary_stats(),
                    top_k=min(5, len(candidate_pool))
                )

        if not ranked_candidates:
            controller.state.status = "COMPLETED"
            controller.state.stopping_reason = "No valid candidate hypotheses remaining in pool."
            print(f"\n[AUTONOMOUS STOP]: {controller.state.stopping_reason}")
            break

        # Adapt candidates if they are RefinedHypothesisRecord
        adapted_candidates = []
        for c in ranked_candidates:
            if hasattr(c, "to_hypothesis_record"):
                rec = c.to_hypothesis_record()
                adapted_candidates.append(rec.model_dump())
            elif isinstance(c, HypothesisRecord):
                adapted_candidates.append(c.model_dump())
            else:
                adapted_candidates.append(c)

        # 2. Validate
        val_report = validate_hypotheses(
            hypotheses=adapted_candidates,
            df=df,
            target=target,
            stage1_data=stage1_data,
            stage2_data=stage2_data,
            iteration_no=iteration_no,
            output_dir=out_dir
        )

        # 3. Plan Experiments
        specs = plan_experiments(
            validation_report=val_report,
            target=target,
            df=df,
            iteration_no=iteration_no,
            output_dir=out_dir
        )

        # 4. Filter duplicates and invalid specifications
        valid_specs: List[ExperimentSpecRecord] = []
        duplicate_candidates = []
        for s, cand in zip(specs, ranked_candidates):
            # Check controller validation
            is_valid, msg = controller.validate_experiment_spec(s)
            if not is_valid:
                print(f"[CONTROLLER REJECTED] {s.experiment_id}: {msg}")
                continue

            # Check duplicate in memory
            if memory.is_duplicate_experiment(s):
                print(f"[CONTROLLER DUPLICATE DETECTED] {s.experiment_id} ({s.features}) already executed. Skipping.")
                duplicate_candidates.append(cand)
                continue

            memory.register_experiment_spec(s)
            valid_specs.append(s)

        if not valid_specs:
            print("[CONTROLLER]: All planned specifications were duplicates or invalid.")
            # Remove duplicate candidates from candidate pool
            candidate_pool = [c for c in candidate_pool if c not in duplicate_candidates]
            controller.state.iterations_completed += 1

            if not candidate_pool:
                # Prompt LLM for novel alternative experiments
                alternatives = []
                for dup in duplicate_candidates[:2]:
                    alt = prompt_llm_for_alternative_experiment(
                        rejected_hypothesis=dup,
                        rejection_reason="Duplicate experiment configuration already executed.",
                        valid_columns=valid_columns
                    )
                    alt_feats = getattr(alt, "features_involved", None) or getattr(alt, "features", None) if alt else None
                    if alt and alt_feats:
                        exp_type = "individual_dependency" if len(alt_feats) <= 1 else "feature_combination_synergy"
                        temp_spec = ExperimentSpecRecord(
                            experiment_id="temp",
                            hypothesis_id=alt.hypothesis_id,
                            iteration_no=iteration_no,
                            experiment_type=exp_type,
                            method="paired_model_comparison" if len(alt_feats) > 1 else "logistic_regression",
                            model_type=(alt.proposed_experiment.get("model") if isinstance(alt.proposed_experiment, dict) else getattr(alt.proposed_experiment, "model", None)) or "sklearn.linear_model.LogisticRegression",
                            features=alt_feats,
                            target=target
                        )
                        if not memory.is_duplicate_experiment(temp_spec):
                            alternatives.append(alt)
                if alternatives:
                    print(f"[CONTROLLER]: LLM proposed {len(alternatives)} novel unexplored candidates.")
                    candidate_pool = alternatives
                else:
                    controller.state.status = "COMPLETED"
                    controller.state.stopping_reason = "All planned candidate configurations have been explored (duplicate exhaustion)."
                    print(f"\n[AUTONOMOUS STOP]: {controller.state.stopping_reason}")
                    break

            iteration_no += 1
            continue

        # 5. Execute Experiments
        results = execute_all_experiments(
            specs=valid_specs,
            df=df,
            iteration_no=iteration_no,
            output_dir=out_dir,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap
        )

        # 6. Evaluate Results
        eval_results, comparison = evaluate_all_results(
            results=results,
            iteration_no=iteration_no,
            output_dir=out_dir
        )

        # Record in memory & update progress
        for r in eval_results:
            memory.save_experiment_result(r)
            controller.update_progress(r)
            print(f"Progress: Experiments={controller.state.experiments_completed}/{cfg.max_experiments} | Current Best Score={controller.state.best_score:.4f}")

        controller.state.iterations_completed += 1

        # Check stopping conditions immediately after experiment batch
        should_stop, stop_reason = controller.should_stop()
        if should_stop:
            controller.state.status = "COMPLETED"
            controller.state.stopping_reason = stop_reason
            print(f"\n[AUTONOMOUS STOPPING CONDITION TRIGGERED]: {stop_reason}")
            break

        # 7. LLM Interpretation & Refinement
        interps, refined_hypotheses = interpret_and_refine_all(
            eval_results=eval_results,
            hypotheses=adapted_candidates,
            iteration_no=iteration_no,
            output_dir=out_dir,
            registry=registry,
            available_features=valid_columns
        )

        # 8. Refill candidate pool for next iteration
        candidate_pool = list(refined_hypotheses)
        iteration_no += 1

    # Generate Final Report
    report_md, report_data = controller.generate_final_report()
    print("\n" + "=" * 70)
    print("AUTONOMOUS SCIENTIFIC DISCOVERY COMPLETE")
    print(f"Final Report Written: {os.path.join(out_dir, 'final_scientific_report.md')}")
    print("=" * 70)
    return report_md, report_data
