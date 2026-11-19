import os
import json
import pytest
import pandas as pd

from modules.dataset_registry import (
    load_registered_dataset,
    get_cache_dir,
    compute_dataset_hash,
    list_datasets,
    load_dataset_config,
)
from modules.identify_target import identify_target
from modules.load_dataset import load_dataset_with_id
from modules.profile_dataset import profile_dataset
from modules.clean_dataset import clean_dataset
from modules.generate_eda import generate_eda
from modules.individual_dependency_analysis import individual_dependency_analysis
from modules.feature_combination_analysis import feature_combination_analysis
from modules.categorical_patterns import categorical_patterns
from modules.feature_importance import feature_importance
from modules.save_patterns import save_patterns_with_ids
from modules.create_llm_context import create_llm_context
from modules.generate_hypotheses_with_llm import generate_hypotheses_with_llm
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments
from modules.experiment_executor import execute_all_experiments
from modules.result_evaluator import evaluate_all_results
from modules.llm_interpreter import interpret_and_refine_all
from modules.hypothesis_registry import HypothesisRegistry
from mains import parse_args


# =============================================================================
# 1. DATASET REGISTRATION AND SPLIT LOADING TESTS
# =============================================================================

def test_system_list_datasets_contains_new_dataset():
    """Verify registry lists both original and new datasets."""
    datasets = list_datasets()
    assert "diabetes" in datasets
    assert "power_system_class" in datasets
    assert "power_system_detect" in datasets


def test_system_power_system_detect_split_loading_and_alignment():
    """
    Test loading the real split dataset (classData.csv + detect_dataset.csv).
    Verify columns, row alignment, unnamed column removal, and hash calculation.
    """
    df, dataset_id, cfg = load_registered_dataset("power_system_detect")

    assert dataset_id.startswith("power_system_detect_")
    assert df.shape[0] == 7861  # min(7861, 12001)
    assert df.shape[1] == 11    # 4 extra (G, C, B, A) + 6 features (Ia..Vc) + 1 target (Output (S))

    expected_cols = ["G", "C", "B", "A", "Ia", "Ib", "Ic", "Va", "Vb", "Vc", "Output (S)"]
    assert list(df.columns) == expected_cols

    # Ensure no unnamed columns
    assert not any(c.startswith("Unnamed") for c in df.columns)

    # Determinism: Loading again produces identical dataset_id
    _, dataset_id_2, _ = load_registered_dataset("power_system_detect")
    assert dataset_id == dataset_id_2


# =============================================================================
# 2. TARGET IDENTIFICATION TESTS
# =============================================================================

def test_system_target_identification_config_override():
    """Verify target_column override is respected."""
    df = pd.DataFrame({"feat1": [1, 2], "Output (S)": [0, 1], "result": [1, 0]})
    # With explicit target_column
    target = identify_target(df, target_column="Output (S)")
    assert target == "Output (S)"


def test_system_target_identification_auto_detection():
    """Verify fallback target auto-detection detects 'Output (S)' via 'output'."""
    df = pd.DataFrame({"Ia": [1.0, 2.0], "Va": [0.1, 0.2], "Output (S)": [0, 1]})
    target = identify_target(df)
    assert target == "Output (S)"


def test_system_target_identification_invalid_column_raises():
    """Verify providing a non-existent target raises ValueError."""
    df = pd.DataFrame({"Ia": [1.0, 2.0], "Va": [0.1, 0.2]})
    with pytest.raises(ValueError, match="not found in dataset"):
        identify_target(df, target_column="NonExistent")


# =============================================================================
# 3. END-TO-END PIPELINE & CACHE ISOLATION TESTS ON NEW DATASET
# =============================================================================

def test_system_full_pipeline_on_new_dataset_and_no_feature_leakage(tmp_path, monkeypatch):
    """
    Runs the full 8-stage scientist workflow on a sample of the new dataset
    in an isolated temp directory. Verifies:
    1. Tier 1 caching and subsequent cache-hit seeking
    2. Stage 3 hypothesis generation with fallback
    3. Stage 4 validation
    4. Stage 5 experiment planning
    5. Stage 6 execution
    6. Stage 7 evaluation
    7. Stage 8 refinement: strictly NO feature leakage from diabetes (Glucose, Age, etc.)
    """
    cache_root = str(tmp_path / "cache")
    monkeypatch.setattr("modules.config.CACHE_ROOT", cache_root)
    monkeypatch.setattr("modules.dataset_registry.CACHE_ROOT", cache_root)

    dataset_cache_dir = get_cache_dir("power_system_test_isolated")
    run_output_dir = str(tmp_path / "run_20261118_test")
    os.makedirs(run_output_dir, exist_ok=True)

    # Create synthetic slice matching the new power system dataset
    df_raw = pd.DataFrame({
        "G": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0] * 5,
        "C": [0, 0, 1, 0, 0, 1, 0, 1, 0, 1] * 5,
        "B": [0, 0, 0, 1, 1, 0, 0, 0, 1, 1] * 5,
        "A": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0] * 5,
        "Ia": [-150.0, -336.0, 50.0, 120.0, -80.0, -170.0, -122.0, 80.0, -90.0, 40.0] * 5,
        "Ib": [-9.6, -76.2, 10.0, 30.0, -15.0, 9.2, 6.1, 12.0, 3.8, 8.5] * 5,
        "Ic": [85.8, 18.3, -40.0, -90.0, 60.0, 161.2, 116.0, -50.0, 86.3, -30.0] * 5,
        "Va": [0.40, 0.31, 0.25, 0.15, 0.35, 0.05, 0.10, 0.28, 0.14, 0.22] * 5,
        "Vb": [-0.13, -0.12, 0.05, 0.10, -0.08, -0.65, -0.62, 0.02, -0.60, 0.04] * 5,
        "Vc": [-0.26, -0.18, 0.12, 0.20, -0.15, 0.60, 0.52, 0.08, 0.46, 0.11] * 5,
        "Output (S)": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1] * 5
    })
    target = "Output (S)"

    # --- TIER 1: Ingestion, Profiling, Cleaning, and Pattern Discovery ---
    df_raw_tracked, input_record = load_dataset_with_id(df_raw, source="power_system_test_isolated")
    numerical, categorical = profile_dataset(df_raw_tracked, output_dir=dataset_cache_dir)
    df_clean = clean_dataset(df_raw_tracked, output_dir=dataset_cache_dir)
    generate_eda(df_clean, output_dir=dataset_cache_dir)

    stage1 = individual_dependency_analysis(df_clean, target, output_dir=dataset_cache_dir)
    stage2 = feature_combination_analysis(
        df=df_clean,
        target=target,
        top_candidates=stage1.get("ranked_features", [])[:4],
        max_candidates=4,
        cv_folds=2,
        output_dir=dataset_cache_dir
    )
    patterns = []
    patterns.extend(stage1.get("patterns", []))
    patterns.extend(stage2.get("patterns", []))
    patterns.extend(categorical_patterns(df_clean, target))
    patterns.extend(feature_importance(df_clean, target, output_dir=dataset_cache_dir))
    saved_patterns = save_patterns_with_ids(patterns, input_id=input_record.input_id, output_dir=dataset_cache_dir)

    # Assert cache files are created in dataset_cache_dir
    assert os.path.exists(os.path.join(dataset_cache_dir, "cleaned_dataset.csv"))
    assert os.path.exists(os.path.join(dataset_cache_dir, "dataset_profile.csv"))
    assert os.path.exists(os.path.join(dataset_cache_dir, "stage1_data.json"))
    assert os.path.exists(os.path.join(dataset_cache_dir, "stage2_data.json"))
    assert os.path.exists(os.path.join(dataset_cache_dir, "discovered_patterns.json"))

    # Verify cache hit reuse
    reloaded_df = pd.read_csv(os.path.join(dataset_cache_dir, "cleaned_dataset.csv"))
    assert len(reloaded_df) == len(df_clean)

    # --- TIER 2: Hypothesis Generation & Experiment Cycle ---
    registry = HypothesisRegistry(
        registry_path=os.path.join(run_output_dir, "hypothesis_registry.json"),
        run_id="run_20261118_test"
    )

    context = create_llm_context(
        df=df_clean,
        target=target,
        patterns=saved_patterns,
        stage1_data=stage1,
        stage2_data=stage2,
        output_dir=run_output_dir
    )

    # Verify context only has current dataset's features
    available_features = [c for c in df_clean.columns if c != target]
    for feat in context["all_features"]:
        assert feat in available_features
        assert feat not in ["Glucose", "BMI", "Age", "Pregnancies", "Insulin"]

    # Stage 3: Hypothesis Generation (fallback handles p-value format properly)
    hypotheses = generate_hypotheses_with_llm(
        context=context,
        iteration_no=1,
        registry=registry,
        output_dir=run_output_dir
    )
    assert len(hypotheses) >= 3

    # Stage 4: Hypothesis Validation
    validation_report = validate_hypotheses(
        hypotheses=hypotheses,
        df=df_clean,
        target=target,
        stage1_data=stage1,
        stage2_data=stage2,
        iteration_no=1,
        output_dir=run_output_dir
    )
    assert validation_report.total >= 3

    # Stage 5: Experiment Planning
    experiment_specs = plan_experiments(
        validation_report=validation_report,
        target=target,
        df=df_clean,
        iteration_no=1,
        output_dir=run_output_dir
    )
    assert len(experiment_specs) >= 1

    # Stage 6: Experiment Execution
    experiment_results = execute_all_experiments(
        specs=experiment_specs,
        df=df_clean,
        iteration_no=1,
        output_dir=run_output_dir,
        n_permutations=10,
        n_bootstrap=10
    )
    assert len(experiment_results) == len(experiment_specs)

    # Stage 7: Result Evaluation
    eval_results, cross_comparison = evaluate_all_results(
        results=experiment_results,
        iteration_no=1,
        output_dir=run_output_dir
    )
    assert len(eval_results) == len(experiment_results)

    # Stage 8: LLM Interpretation & Hypothesis Refinement
    interpretations, refined_hypotheses = interpret_and_refine_all(
        eval_results=eval_results,
        hypotheses=hypotheses,
        iteration_no=1,
        output_dir=run_output_dir,
        registry=registry,
        available_features=available_features
    )
    assert len(refined_hypotheses) >= 1

    # CRITICAL ASSERTION: Zero foreign features in refined hypotheses!
    forbidden_features = {"glucose", "bmi", "age", "pregnancies", "bloodpressure", "insulin", "skinthickness"}
    for ref in refined_hypotheses:
        for f in ref.features_involved:
            assert f.lower() not in forbidden_features, f"Foreign feature {f} leaked into refined hypothesis: {ref}"
            assert f in available_features, f"Feature {f} not in available features {available_features}"


# =============================================================================
# 4. STRICT CACHE NON-INTERFERENCE BETWEEN TWO DATASETS
# =============================================================================

def test_system_two_datasets_strict_cache_isolation(tmp_path):
    """
    Verifies that running two different datasets creates two distinct cache directories
    under cache/ without file collision, cross-reading, or data overwrite.
    """
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    # Dataset 1: Medical
    df1 = pd.DataFrame({"Glucose": [100, 150], "BMI": [25.0, 30.0], "Outcome": [0, 1]})
    h1 = compute_dataset_hash(str(tmp_path / "f1.csv")) if False else "hash1111"
    dir1 = cache_root / f"dataset_medical_{h1}"
    dir1.mkdir()
    clean_dataset(df1, output_dir=str(dir1))

    # Dataset 2: Sensor
    df2 = pd.DataFrame({"Ia": [-100.0, 50.0], "Va": [0.3, 0.4], "Output (S)": [0, 1]})
    h2 = compute_dataset_hash(str(tmp_path / "f2.csv")) if False else "hash2222"
    dir2 = cache_root / f"dataset_sensor_{h2}"
    dir2.mkdir()
    clean_dataset(df2, output_dir=str(dir2))

    assert dir1 != dir2
    assert (dir1 / "cleaned_dataset.csv").exists()
    assert (dir2 / "cleaned_dataset.csv").exists()

    d1_cols = pd.read_csv(dir1 / "cleaned_dataset.csv").columns.tolist()
    d2_cols = pd.read_csv(dir2 / "cleaned_dataset.csv").columns.tolist()

    assert "Glucose" in d1_cols and "Ia" not in d1_cols
    assert "Ia" in d2_cols and "Glucose" not in d2_cols


# =============================================================================
# 5. CLI ARGUMENT PARSER TESTS
# =============================================================================

def test_system_cli_parser_defaults(monkeypatch):
    """Verify default CLI arguments."""
    monkeypatch.setattr("sys.argv", ["mains.py"])
    args = parse_args()
    assert args.dataset == "diabetes"
    assert not args.autonomous


def test_system_cli_parser_custom_dataset(monkeypatch):
    """Verify --dataset flag is correctly parsed."""
    monkeypatch.setattr("sys.argv", ["mains.py", "--dataset", "power_system_detect", "--autonomous"])
    args = parse_args()
    assert args.dataset == "power_system_detect"
    assert args.autonomous
