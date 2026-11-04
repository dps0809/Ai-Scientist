import os
import json
import pytest
import pandas as pd
from modules.profile_dataset import profile_dataset
from modules.clean_dataset import clean_dataset
from modules.generate_eda import generate_eda
from modules.individual_dependency_analysis import individual_dependency_analysis
from modules.feature_combination_analysis import feature_combination_analysis
from modules.feature_importance import feature_importance
from modules.save_patterns import save_patterns_with_ids

def test_dataset_processed_caching_and_seeking(tmp_path):
    dataset_cache_dir = tmp_path / "dataset_processed"
    dataset_cache_dir.mkdir()
    
    # Create sample dataset
    df_raw = pd.DataFrame({
        "Glucose": [100, 120, 140, 160, 180, 200],
        "BMI": [25.0, 28.0, 31.0, 34.0, 37.0, 40.0],
        "Age": [22, 28, 35, 42, 50, 60],
        "Outcome": [0, 0, 0, 1, 1, 1]
    })
    
    # Run 1: Cache is empty -> Computes and saves to dataset_processed
    numerical, categorical = profile_dataset(df_raw, output_dir=str(dataset_cache_dir))
    df_cleaned = clean_dataset(df_raw, output_dir=str(dataset_cache_dir))
    generate_eda(df_cleaned, output_dir=str(dataset_cache_dir))
    
    stage1 = individual_dependency_analysis(df_cleaned, target="Outcome", output_dir=str(dataset_cache_dir))
    stage2 = feature_combination_analysis(
        df=df_cleaned,
        target="Outcome",
        top_candidates=["Glucose", "BMI", "Age"],
        max_candidates=3,
        cv_folds=2,
        output_dir=str(dataset_cache_dir)
    )
    feat_imp = feature_importance(df_cleaned, target="Outcome", output_dir=str(dataset_cache_dir))
    
    patterns = []
    patterns.extend(stage1["patterns"])
    patterns.extend(stage2["patterns"])
    patterns.extend(feat_imp)
    saved_patterns = save_patterns_with_ids(patterns, input_id="INP-TEST", output_dir=str(dataset_cache_dir))
    
    # Assert all dataset-level files exist in dataset_processed/
    assert (dataset_cache_dir / "dataset_profile.csv").exists()
    assert (dataset_cache_dir / "cleaned_dataset.csv").exists()
    assert (dataset_cache_dir / "correlation_matrix.csv").exists()
    assert (dataset_cache_dir / "individual_dependencies.csv").exists()
    assert (dataset_cache_dir / "stage1_data.json").exists()
    assert (dataset_cache_dir / "feature_combinations.csv").exists()
    assert (dataset_cache_dir / "stage2_data.json").exists()
    assert (dataset_cache_dir / "feature_importance.csv").exists()
    assert (dataset_cache_dir / "discovered_patterns.json").exists()
    
    # Run 2: Cache exists -> Seek and reuse directly (fast cache hit)
    reloaded_df = clean_dataset(df_raw, output_dir=str(dataset_cache_dir))
    assert len(reloaded_df) == len(df_cleaned)
    
    cached_stage1 = individual_dependency_analysis(df_cleaned, target="Outcome", output_dir=str(dataset_cache_dir))
    assert cached_stage1["ranked_features"] == stage1["ranked_features"]
    
    cached_stage2 = feature_combination_analysis(
        df=df_cleaned,
        target="Outcome",
        top_candidates=["Glucose", "BMI", "Age"],
        output_dir=str(dataset_cache_dir)
    )
    assert len(cached_stage2["combinations"]) == len(stage2["combinations"])
    
    cached_patterns = save_patterns_with_ids([], input_id="INP-TEST", output_dir=str(dataset_cache_dir))
    assert len(cached_patterns) == len(saved_patterns)
