import os
import sys
import json
import warnings
import argparse
from datetime import datetime
import pandas as pd

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

# pyrefly: ignore [missing-import]
from modules.config import (
    DATASET_PATH,
    OUTPUT_DIR,
    DATASET_PROCESSED_DIR,
    CACHE_ROOT,
    get_dataset_cache_dir,
    TOP_K_CANDIDATES,
    CV_FOLDS,
    VALIDATED_HYPOTHESES_PATH,
    EXPERIMENT_SPECS_PATH,
    DB_INPUTS,
    DB_PATTERNS,
    DB_HYPOTHESES,
    DB_VALIDATION,
    DB_EXPERIMENTS,
    DB_RESULTS,
    DB_EVALUATIONS,
    DB_REFINEMENTS,
    DB_MEMORY,
    MAX_AUTONOMOUS_EXPERIMENTS,
    MAX_AUTONOMOUS_ITERATIONS,
    MAX_RUNTIME_MINUTES
)
# pyrefly: ignore [missing-import]
from modules.dataset_registry import (
    load_registered_dataset,
    get_cache_dir,
    list_datasets
)
# pyrefly: ignore [missing-import]
from modules.load_dataset import load_dataset_with_id
# pyrefly: ignore [missing-import]
from modules.profile_dataset import profile_dataset
# pyrefly: ignore [missing-import]
from modules.clean_dataset import clean_dataset
# pyrefly: ignore [missing-import]
from modules.generate_eda import generate_eda
# pyrefly: ignore [missing-import]
from modules.identify_target import identify_target
# pyrefly: ignore [missing-import]
from modules.individual_dependency_analysis import individual_dependency_analysis
# pyrefly: ignore [missing-import]
from modules.feature_combination_analysis import feature_combination_analysis
# pyrefly: ignore [missing-import]
from modules.categorical_patterns import categorical_patterns
# pyrefly: ignore [missing-import]
from modules.feature_importance import feature_importance
# pyrefly: ignore [missing-import]
from modules.save_patterns import save_patterns_with_ids
# pyrefly: ignore [missing-import]
from modules.create_llm_context import create_llm_context
# pyrefly: ignore [missing-import]
from modules.generate_hypotheses_with_llm import generate_hypotheses_with_llm
# pyrefly: ignore [missing-import]
from modules.hypothesis_validator import validate_hypotheses
# pyrefly: ignore [missing-import]
from modules.experiment_planner import plan_experiments
# pyrefly: ignore [missing-import]
from modules.experiment_executor import execute_all_experiments
# pyrefly: ignore [missing-import]
from modules.result_evaluator import evaluate_all_results
# pyrefly: ignore [missing-import]
from modules.llm_interpreter import interpret_and_refine_all
# pyrefly: ignore [missing-import]
from modules.hypothesis_registry import HypothesisRegistry
# pyrefly: ignore [missing-import]
from modules.experiment_controller import run_autonomous_scientist
# pyrefly: ignore [missing-import]
from modules.schemas import StoppingConditionConfig
# pyrefly: ignore [missing-import]
from modules.mongo_cluster import MongoClusterManager

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Scientist: Closed-Loop Data-Driven Hypothesis & Experiment Engine"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="diabetes",
        help="Name of dataset from configs/datasets.toml (default: diabetes)"
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Run in closed-loop autonomous multi-iteration mode (Batches 6 & 7)"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=MAX_AUTONOMOUS_EXPERIMENTS,
        help=f"Maximum total experiments budget (default: {MAX_AUTONOMOUS_EXPERIMENTS})"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_AUTONOMOUS_ITERATIONS,
        help=f"Maximum autonomous iterations (default: {MAX_AUTONOMOUS_ITERATIONS})"
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=int,
        default=MAX_RUNTIME_MINUTES,
        help=f"Maximum allowed runtime in minutes (default: {MAX_RUNTIME_MINUTES})"
    )
    parser.add_argument(
        "--target-auc",
        type=float,
        default=0.85,
        help="Target ROC-AUC stopping threshold (default: 0.85)"
    )
    parser.add_argument(
        "--no-improvement-tolerance",
        type=int,
        default=3,
        help="Consecutive iterations without gain tolerance (default: 3)"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutation test iterations (default: 1000)"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Number of bootstrap iterations for CI (default: 500)"
    )
    parser.add_argument(
        "--no-mongo",
        action="store_true",
        help="Disable MongoDB synchronization (run purely file-based)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n")
    print("=" * 70)
    print("              LIGHTWEIGHT AI SCIENTIST")
    print("    ITERATIVE DATA-DRIVEN HYPOTHESIS & EXPERIMENT ENGINE")
    if args.autonomous:
        print("          [MODE: AUTONOMOUS CLOSED-LOOP SCIENTIST]")
    else:
        print("              [MODE: STANDARD SINGLE-RUN]")
    print("=" * 70)

    # =========================================================================
    # TIER 1: DATASET PROCESSING LAYER (Isolated per dataset via Content Hash)
    # Stored in: ai_scientist_output/cache/<dataset_id>/
    # Seek existing processed data first; compute and store only if not found.
    # =========================================================================
    df_raw, dataset_id, ds_config = load_registered_dataset(args.dataset)
    dataset_cache_dir = get_cache_dir(dataset_id)
    target_override = ds_config.get("output_target") or ds_config.get("target_column")

    cleaned_csv_path = os.path.join(dataset_cache_dir, "cleaned_dataset.csv")
    profile_csv_path = os.path.join(dataset_cache_dir, "dataset_profile.csv")
    stage1_json_path = os.path.join(dataset_cache_dir, "stage1_data.json")
    stage2_json_path = os.path.join(dataset_cache_dir, "stage2_data.json")
    patterns_json_path = os.path.join(dataset_cache_dir, "discovered_patterns.json")

    has_cached_dataset = (
        os.path.exists(cleaned_csv_path)
        and os.path.exists(profile_csv_path)
        and os.path.exists(stage1_json_path)
        and os.path.exists(stage2_json_path)
        and os.path.exists(patterns_json_path)
    )

    if has_cached_dataset:
        print("\n" + "=" * 70)
        print("DATASET PROCESSING LAYER: REUSING CACHED PROCESSED DATASET")
        print(f"Dataset ID : {dataset_id}")
        print(f"Location   : {dataset_cache_dir}")
        print("=" * 70)
        df = pd.read_csv(cleaned_csv_path)
        target = identify_target(df, target_column=target_override)
        with open(stage1_json_path, "r") as f:
            stage1 = json.load(f)
        with open(stage2_json_path, "r") as f:
            stage2 = json.load(f)
        with open(patterns_json_path, "r") as f:
            patterns = json.load(f)
        print(f"[CACHE HIT] Cleaned dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"[CACHE HIT] Stage 1 individual dependencies loaded: {len(stage1.get('ranked_features', []))} features")
        print(f"[CACHE HIT] Stage 2 combinations & interactions loaded: {len(stage2.get('combinations', []))} pairs")
        print(f"[CACHE HIT] Discovered patterns loaded: {len(patterns)} patterns")
    else:
        print("\n" + "=" * 70)
        print("DATASET PROCESSING LAYER: COMPUTING & PERSISTING FOR FUTURE RUNS")
        print(f"Dataset ID  : {dataset_id}")
        print(f"Destination : {dataset_cache_dir}")
        print("=" * 70)
        # Ingestion & Profiling Layer with Unique Input Tracking
        df_raw, input_record = load_dataset_with_id(df_raw, source=dataset_id)
        numerical, categorical = profile_dataset(df_raw, output_dir=dataset_cache_dir)
        df = clean_dataset(df_raw, output_dir=dataset_cache_dir)
        generate_eda(df, output_dir=dataset_cache_dir)
        target = identify_target(df, target_column=target_override)

        print(f"\nSelected target variable: {target}")

        patterns = []

        # Stage 1: Individual Feature Dependency Analysis
        stage1 = individual_dependency_analysis(df, target, output_dir=dataset_cache_dir)
        patterns.extend(stage1.get("patterns", []))

        # Stage 2: Feature Combination Discovery & Interaction Analysis
        stage2 = feature_combination_analysis(
            df=df,
            target=target,
            top_candidates=stage1.get("ranked_features", []),
            max_candidates=TOP_K_CANDIDATES,
            cv_folds=CV_FOLDS,
            output_dir=dataset_cache_dir
        )
        patterns.extend(stage2.get("patterns", []))

        # Complementary Categorical and Tree Feature Importance signals
        patterns.extend(categorical_patterns(df, target))
        patterns.extend(feature_importance(df, target, output_dir=dataset_cache_dir))

        # Save Patterns with unique pattern IDs and link to input_id
        patterns = save_patterns_with_ids(patterns, input_id=input_record.input_id, output_dir=dataset_cache_dir)

    # =========================================================================
    # TIER 2: RUN HYPOTHESIS & EXPERIMENT LAYER (Timestamped Run Directory)
    # Stored in: ai_scientist_output/<YYYYMMDD_HHMMSS>/
    # =========================================================================
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    registry = HypothesisRegistry(
        registry_path=os.path.join(run_output_dir, "hypothesis_registry.json"),
        run_id=run_timestamp
    )

    print("\n" + "=" * 70)
    print(f"RUN HYPOTHESIS & EXPERIMENT CYCLE")
    print(f"RUN TIMESTAMP: {run_timestamp}")
    print(f"RUN DIRECTORY: {run_output_dir}")
    print("=" * 70)

    # Build LLM Context
    context = create_llm_context(
        df=df,
        target=target,
        patterns=patterns,
        stage1_data=stage1,
        stage2_data=stage2,
        output_dir=run_output_dir
    )

    # Stage 3: LLM Hypothesis Generation & De-duplication (Iteration 1)
    iteration_no = 1
    hypotheses = generate_hypotheses_with_llm(context, iteration_no=iteration_no, registry=registry, output_dir=run_output_dir)

    # Check MongoDB cluster sync status
    cluster = MongoClusterManager()
    cluster_status = "CONNECTED (Port 27017)" if cluster.is_available() and not args.no_mongo else "STANDALONE FILE-BASED (Offline)"

    if args.autonomous:
        # =====================================================================
        # BATCH 6 & 7: AUTONOMOUS CLOSED-LOOP EXECUTION
        # =====================================================================
        stopping_cfg = StoppingConditionConfig(
            max_experiments=args.max_experiments,
            max_iterations=args.max_iterations,
            max_runtime_minutes=args.max_runtime_minutes,
            no_improvement_tolerance=args.no_improvement_tolerance,
            target_roc_auc=args.target_auc
        )
        print("\n" + "=" * 70)
        print("LAUNCHING BATCH 6 & 7: AUTONOMOUS DISCOVERY CONTROLLER")
        print("=" * 70)
        report_md, report_data = run_autonomous_scientist(
            initial_hypotheses=hypotheses,
            df=df,
            target=target,
            config=stopping_cfg,
            output_dir=run_output_dir,
            stage1_data=stage1,
            stage2_data=stage2,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            sync_mongo=not args.no_mongo
        )

        print("\n" + "=" * 70)
        print("AUTONOMOUS DISCOVERY CYCLE COMPLETE")
        print("=" * 70)
        print(f"Total Iterations Completed : {report_data.get('total_iterations')}")
        print(f"Total Experiments Executed : {report_data.get('total_experiments')}")
        print(f"Best ROC-AUC Achieved      : {report_data.get('best_roc_auc', 0.0):.4f}")
        print(f"Best Model Architecture    : {report_data.get('best_model')}")
        print(f"Best Feature Combination   : {report_data.get('best_features') or report_data.get('best_feature_combination')}")
        print(f"\nFinal Scientific Report Generated:")
        print(f"  Markdown: {os.path.join(run_output_dir, 'final_scientific_report.md')}")
        print(f"  JSON    : {os.path.join(run_output_dir, 'final_scientific_report.json')}")
        print(f"  Memory  : {os.path.join(run_output_dir, 'experiment_memory.json')}")
        print(f"  State   : {os.path.join(run_output_dir, 'autonomous_state.json')}")

    else:
        # =====================================================================
        # STANDARD SINGLE-RUN (ITERATION 1) EXECUTION (Batches 1 - 5)
        # =====================================================================
        # Stage 4: Batch 1 — Hypothesis Validation & Evidence Reconciliation
        validation_report = validate_hypotheses(
            hypotheses=hypotheses,
            df=df,
            target=target,
            stage1_data=stage1,
            stage2_data=stage2,
            iteration_no=iteration_no,
            output_dir=run_output_dir
        )

        # Stage 5: Batch 2 — Experiment Registry Planning & Specifications
        experiment_specs = plan_experiments(
            validation_report=validation_report,
            target=target,
            df=df,
            iteration_no=iteration_no,
            output_dir=run_output_dir
        )

        # Stage 6: Batch 3 — Experiment Execution Engine
        experiment_results = execute_all_experiments(
            specs=experiment_specs,
            df=df,
            iteration_no=iteration_no,
            output_dir=run_output_dir,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap
        )

        # Stage 7: Batch 4 — Result Evaluation & Cross-Experiment Comparison
        eval_results, cross_comparison = evaluate_all_results(
            results=experiment_results,
            iteration_no=iteration_no,
            output_dir=run_output_dir
        )

        # Stage 8: Batch 5 — LLM Interpretation & Hypothesis Refinement Engine
        interpretations, refined_hypotheses = interpret_and_refine_all(
            eval_results=eval_results,
            hypotheses=hypotheses,
            iteration_no=iteration_no,
            output_dir=run_output_dir,
            registry=registry,
            available_features=[c for c in df.columns if c != target]
        )

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print("\nCurrent implemented flow:")
        print("Processed Dataset (Cache-First) -> Stage 3 (LLM Hypotheses) -> Stage 4 (Batch 1 Validation) -> Stage 5 (Batch 2 Planning) -> Stage 6 (Batch 3 Execution) -> Stage 7 (Batch 4 Evaluation) -> Stage 8 (Batch 5 Refinement)")
        print(f"\nRun-specific hypothesis & experiment outputs:")
        print(f"  {run_output_dir}/")
        print(f"  - {run_output_dir}/llm_context.json")
        print(f"  - {run_output_dir}/generated_hypotheses.json")
        print(f"  - {run_output_dir}/validated_hypotheses.json")
        print(f"  - {run_output_dir}/experiment_specifications.json")
        print(f"  - {run_output_dir}/experiment_results.json")
        print(f"  - {run_output_dir}/experiment_evaluations.json")
        print(f"  - {run_output_dir}/cross_experiment_comparison.json")
        print(f"  - {run_output_dir}/llm_interpretations.json")
        print(f"  - {run_output_dir}/refined_next_hypotheses.json")
        print(f"  - {run_output_dir}/iterations/iter_{iteration_no}/")

    print(f"\nMongoDB Cluster Status: {cluster_status}")
    if cluster.is_available() and not args.no_mongo:
        print(f"  - Inputs DB       : {DB_INPUTS}.inputs")
        print(f"  - Patterns DB     : {DB_PATTERNS}.patterns")
        print(f"  - Hypotheses DB   : {DB_HYPOTHESES}.hypotheses")
        print(f"  - Validation DB   : {DB_VALIDATION}.validations")
        print(f"  - Experiments DB  : {DB_EXPERIMENTS}.specifications")
        print(f"  - Results DB      : {DB_RESULTS}.results")
        print(f"  - Evaluations DB  : {DB_EVALUATIONS}.evaluations")
        print(f"  - Refinements DB  : {DB_REFINEMENTS}.interpretations & refined_hypotheses")
        print(f"  - Memory DB       : {DB_MEMORY}.experiments & fingerprints")

    print(f"\nDataset-level processed artifacts (reused across runs):")
    print(f"  {dataset_cache_dir}/")
    print(f"  - {dataset_cache_dir}/cleaned_dataset.csv")
    print(f"  - {dataset_cache_dir}/dataset_profile.csv")
    print(f"  - {dataset_cache_dir}/correlation_matrix.csv")
    print(f"  - {dataset_cache_dir}/individual_dependencies.csv")
    print(f"  - {dataset_cache_dir}/feature_combinations.csv")
    print(f"  - {dataset_cache_dir}/feature_importance.csv")
    print(f"  - {dataset_cache_dir}/discovered_patterns.json")
    print(f"  - {dataset_cache_dir}/plots/")


if __name__ == "__main__":
    main()

