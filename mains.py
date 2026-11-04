import os
import json
import warnings
from datetime import datetime
import pandas as pd

# pyrefly: ignore [missing-import]
from modules.config import (
    DATASET_PATH,
    OUTPUT_DIR,
    DATASET_PROCESSED_DIR,
    TOP_K_CANDIDATES,
    CV_FOLDS,
    VALIDATED_HYPOTHESES_PATH,
    EXPERIMENT_SPECS_PATH,
    DB_INPUTS,
    DB_PATTERNS,
    DB_HYPOTHESES,
    DB_VALIDATION,
    DB_EXPERIMENTS,
    DB_RESULTS
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
from modules.mongo_cluster import MongoClusterManager

warnings.filterwarnings("ignore")


def main():
    print("\n")
    print("=" * 70)
    print("              LIGHTWEIGHT AI SCIENTIST")
    print("    ITERATIVE DATA-DRIVEN HYPOTHESIS & EXPERIMENT ENGINE")
    print("=" * 70)

    # =========================================================================
    # TIER 1: DATASET PROCESSING LAYER (Shared across all hypothesis runs)
    # Stored in: ai_scientist_output/dataset_processed/
    # Seek existing processed data first; compute and store only if not found.
    # =========================================================================
    cleaned_csv_path = os.path.join(DATASET_PROCESSED_DIR, "cleaned_dataset.csv")
    profile_csv_path = os.path.join(DATASET_PROCESSED_DIR, "dataset_profile.csv")
    stage1_json_path = os.path.join(DATASET_PROCESSED_DIR, "stage1_data.json")
    stage2_json_path = os.path.join(DATASET_PROCESSED_DIR, "stage2_data.json")
    patterns_json_path = os.path.join(DATASET_PROCESSED_DIR, "discovered_patterns.json")

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
        print(f"Location: {DATASET_PROCESSED_DIR}")
        print("=" * 70)
        df = pd.read_csv(cleaned_csv_path)
        target = identify_target(df)
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
        print(f"Destination: {DATASET_PROCESSED_DIR}")
        print("=" * 70)
        # Ingestion & Profiling Layer with Unique Input Tracking
        df_raw, input_record = load_dataset_with_id(DATASET_PATH)
        numerical, categorical = profile_dataset(df_raw, output_dir=DATASET_PROCESSED_DIR)
        df = clean_dataset(df_raw, output_dir=DATASET_PROCESSED_DIR)
        generate_eda(df, output_dir=DATASET_PROCESSED_DIR)
        target = identify_target(df)

        print(f"\nSelected target variable: {target}")

        patterns = []

        # Stage 1: Individual Feature Dependency Analysis
        stage1 = individual_dependency_analysis(df, target, output_dir=DATASET_PROCESSED_DIR)
        patterns.extend(stage1.get("patterns", []))

        # Stage 2: Feature Combination Discovery & Interaction Analysis
        stage2 = feature_combination_analysis(
            df=df,
            target=target,
            top_candidates=stage1.get("ranked_features", []),
            max_candidates=TOP_K_CANDIDATES,
            cv_folds=CV_FOLDS,
            output_dir=DATASET_PROCESSED_DIR
        )
        patterns.extend(stage2.get("patterns", []))

        # Complementary Categorical and Tree Feature Importance signals
        patterns.extend(categorical_patterns(df, target))
        patterns.extend(feature_importance(df, target, output_dir=DATASET_PROCESSED_DIR))

        # Save Patterns with unique pattern IDs and link to input_id
        patterns = save_patterns_with_ids(patterns, input_id=input_record.input_id, output_dir=DATASET_PROCESSED_DIR)

    # =========================================================================
    # TIER 2: RUN HYPOTHESIS & EXPERIMENT LAYER (Timestamped Run Directory)
    # Stored in: ai_scientist_output/<YYYYMMDD_HHMMSS>/
    # =========================================================================
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

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
    hypotheses = generate_hypotheses_with_llm(context, iteration_no=iteration_no, output_dir=run_output_dir)

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
        n_permutations=1000,
        n_bootstrap=500
    )

    # Check MongoDB cluster sync status
    cluster = MongoClusterManager()
    cluster_status = "CONNECTED (Port 27017)" if cluster.is_available() else "STANDALONE FILE-BASED (Offline)"

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nCurrent implemented flow:")
    print("Processed Dataset (Cache-First) -> Stage 3 (Local LLM & De-duplication) -> Stage 4 (Batch 1 Validation) -> Stage 5 (Batch 2 Planning) -> Stage 6 (Batch 3 Execution)")
    print(f"\nMongoDB Cluster Status: {cluster_status}")
    if cluster.is_available():
        print(f"  - Inputs DB       : {DB_INPUTS}.inputs")
        print(f"  - Patterns DB     : {DB_PATTERNS}.patterns")
        print(f"  - Hypotheses DB   : {DB_HYPOTHESES}.hypotheses")
        print(f"  - Validation DB   : {DB_VALIDATION}.validations")
        print(f"  - Experiments DB  : {DB_EXPERIMENTS}.specifications")
        print(f"  - Results DB      : {DB_RESULTS}.results")
    print(f"\nDataset-level processed artifacts (reused across runs):")
    print(f"  {DATASET_PROCESSED_DIR}/")
    print(f"  - {DATASET_PROCESSED_DIR}/cleaned_dataset.csv")
    print(f"  - {DATASET_PROCESSED_DIR}/dataset_profile.csv")
    print(f"  - {DATASET_PROCESSED_DIR}/correlation_matrix.csv")
    print(f"  - {DATASET_PROCESSED_DIR}/individual_dependencies.csv")
    print(f"  - {DATASET_PROCESSED_DIR}/feature_combinations.csv")
    print(f"  - {DATASET_PROCESSED_DIR}/feature_importance.csv")
    print(f"  - {DATASET_PROCESSED_DIR}/discovered_patterns.json")
    print(f"  - {DATASET_PROCESSED_DIR}/plots/")
    print(f"\nRun-specific hypothesis & experiment outputs:")
    print(f"  {run_output_dir}/")
    print(f"  - {run_output_dir}/llm_context.json")
    print(f"  - {run_output_dir}/generated_hypotheses.json")
    print(f"  - {run_output_dir}/validated_hypotheses.json")
    print(f"  - {run_output_dir}/experiment_specifications.json")
    print(f"  - {run_output_dir}/experiment_results.json")
    print(f"  - {run_output_dir}/iterations/iter_{iteration_no}/")


if __name__ == "__main__":
    main()
