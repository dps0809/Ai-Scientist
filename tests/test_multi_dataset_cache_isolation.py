import os
import json
import pytest
import pandas as pd

from modules.dataset_registry import (
    compute_dataset_hash,
    get_cache_dir,
    load_registered_dataset,
    load_dataset_config,
    list_datasets,
)
from modules.config import get_dataset_cache_dir
from modules.profile_dataset import profile_dataset
from modules.clean_dataset import clean_dataset
from modules.individual_dependency_analysis import individual_dependency_analysis
from modules.save_patterns import save_patterns_with_ids


def test_compute_dataset_hash_different_data(tmp_path):
    f1 = tmp_path / "data1.csv"
    f2 = tmp_path / "data2.csv"
    f1.write_text("a,b\n1,2\n3,4\n")
    f2.write_text("a,b\n5,6\n7,8\n")

    h1 = compute_dataset_hash(str(f1))
    h2 = compute_dataset_hash(str(f2))

    assert h1 != h2
    assert len(h1) == 8
    assert len(h2) == 8


def test_compute_dataset_hash_same_data(tmp_path):
    f1 = tmp_path / "copy1.csv"
    f2 = tmp_path / "copy2.csv"
    content = "col1,col2\n10,20\n30,40\n"
    f1.write_text(content)
    f2.write_text(content)

    h1 = compute_dataset_hash(str(f1))
    h2 = compute_dataset_hash(str(f2))

    assert h1 == h2


def test_get_cache_dir_creates_directory(tmp_path, monkeypatch):
    test_cache_root = str(tmp_path / "test_cache")
    monkeypatch.setattr("modules.config.CACHE_ROOT", test_cache_root)
    monkeypatch.setattr("modules.dataset_registry.CACHE_ROOT", test_cache_root)

    cache_dir = get_cache_dir("test_dataset_12345678")
    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "plots"))
    assert "test_dataset_12345678" in cache_dir


def test_cache_isolation_between_two_datasets(tmp_path):
    """
    Verifies that two datasets processed into their respective caches
    maintain complete isolation and do not contaminate or overwrite each other.
    """
    cache_a = tmp_path / "cache" / "dataset_a_11111111"
    cache_b = tmp_path / "cache" / "dataset_b_22222222"
    cache_a.mkdir(parents=True)
    cache_b.mkdir(parents=True)

    # Dataset A (Medical data)
    df_a = pd.DataFrame({
        "Glucose": [90, 130, 160, 200, 110],
        "BMI": [22.0, 29.0, 33.0, 38.0, 25.0],
        "Outcome": [0, 0, 1, 1, 0]
    })
    clean_dataset(df_a, output_dir=str(cache_a))
    profile_dataset(df_a, output_dir=str(cache_a))
    stage1_a = individual_dependency_analysis(df_a, target="Outcome", output_dir=str(cache_a))
    save_patterns_with_ids(stage1_a["patterns"], input_id="INP-A", output_dir=str(cache_a))

    # Dataset B (Electrical sensor data)
    df_b = pd.DataFrame({
        "Ia": [-150.0, -300.0, 50.0, 120.0, -80.0],
        "Va": [0.40, 0.31, 0.25, 0.15, 0.35],
        "Output_Fault": [1, 1, 0, 0, 1]
    })
    clean_dataset(df_b, output_dir=str(cache_b))
    profile_dataset(df_b, output_dir=str(cache_b))
    stage1_b = individual_dependency_analysis(df_b, target="Output_Fault", output_dir=str(cache_b))
    save_patterns_with_ids(stage1_b["patterns"], input_id="INP-B", output_dir=str(cache_b))

    # Isolation Assertions
    assert str(cache_a) != str(cache_b)

    # Check Dataset A cache
    df_a_cached = pd.read_csv(cache_a / "cleaned_dataset.csv")
    assert "Glucose" in df_a_cached.columns
    assert "Ia" not in df_a_cached.columns
    with open(cache_a / "discovered_patterns.json") as f:
        patterns_a = json.load(f)
    features_in_a = [p.get("feature") for p in patterns_a]
    assert "Glucose" in features_in_a
    assert "Ia" not in features_in_a

    # Check Dataset B cache
    df_b_cached = pd.read_csv(cache_b / "cleaned_dataset.csv")
    assert "Ia" in df_b_cached.columns
    assert "Glucose" not in df_b_cached.columns
    with open(cache_b / "discovered_patterns.json") as f:
        patterns_b = json.load(f)
    features_in_b = [p.get("feature") for p in patterns_b]
    assert "Ia" in features_in_b
    assert "Glucose" not in features_in_b


def test_load_registered_dataset_single_file(tmp_path):
    data_file = tmp_path / "single_test.csv"
    data_file.write_text("x1,x2,label\n1,2,0\n3,4,1\n")

    cfg_file = tmp_path / "datasets.toml"
    cfg_file.write_text(f"""
[datasets.test_single]
name = "test_single"
type = "single_file"
path = "{str(data_file).replace(chr(92), '/')}"
target_column = "label"
""")

    df, dataset_id, cfg = load_registered_dataset("test_single", config_path=str(cfg_file))
    assert df.shape == (2, 3)
    assert dataset_id.startswith("test_single_")
    assert cfg["target_column"] == "label"


def test_load_registered_dataset_split_input_output(tmp_path):
    in_file = tmp_path / "in.csv"
    out_file = tmp_path / "out.csv"

    in_file.write_text("G,Ia,Ib,Unnamed: 3\n1,-100,50,NaN\n0,-200,60,NaN\n1,-300,70,NaN\n")
    out_file.write_text("Output (S),Ia,Ib,Unnamed: 5\n0,-100,50,NaN\n1,-200,60,NaN\n")

    cfg_file = tmp_path / "datasets.toml"
    cfg_file.write_text(f"""
[datasets.test_split]
name = "test_split"
type = "split_input_output"
input_path = "{str(in_file).replace(chr(92), '/')}"
output_path = "{str(out_file).replace(chr(92), '/')}"
input_features = ["Ia", "Ib"]
extra_input_columns = ["G"]
output_target = "Output (S)"
drop_unnamed = true
""")

    df, dataset_id, cfg = load_registered_dataset("test_split", config_path=str(cfg_file))
    # Merged rows should align to min(3, 2) = 2
    assert df.shape == (2, 4)
    assert list(df.columns) == ["G", "Ia", "Ib", "Output (S)"]
    assert dataset_id.startswith("test_split_")
    assert cfg["output_target"] == "Output (S)"


def test_llm_refinement_feature_isolation(tmp_path):
    """
    Verifies that hypothesis refinement and empirical fallbacks strictly draw from
    the current dataset's available features and never inject hardcoded features
    (e.g., Glucose, Age, BMI) from previous datasets.
    """
    from modules.schemas import StandardExperimentResult
    from modules.llm_interpreter import _build_empirical_fallback, register_refined_hypotheses

    res = StandardExperimentResult(
        experiment_id="EXP-1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        configuration={"features": ["Ia"], "target": "Output (S)"},
        evaluation={"verdict": "supported_by_current_evidence", "summary": "test"}
    )

    available = ["G", "C", "B", "A", "Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
    interp = _build_empirical_fallback(res, available_features=available)

    # Hypothesis statement must not reference Glucose or BMI
    assert "Glucose" not in interp.next_hypothesis
    assert "BMI" not in interp.next_hypothesis
    assert "Age" not in interp.next_hypothesis

    # Proposed experiment features must only contain available features
    for feat in interp.proposed_experiment.get("features", []):
        assert feat in available

    reg_path = str(tmp_path / "hypo_reg.json")
    from modules.hypothesis_registry import HypothesisRegistry
    reg = HypothesisRegistry(registry_path=reg_path)

    refined = register_refined_hypotheses(
        [interp],
        next_iteration_no=2,
        registry=reg,
        available_features=available
    )
    assert len(refined) == 1
    assert "Glucose" not in refined[0].features_involved
    assert "BMI" not in refined[0].features_involved
    for feat in refined[0].features_involved:
        assert feat in available
