import pytest
import pandas as pd
from modules.load_dataset import load_dataset_with_id, load_dataset
from modules.save_patterns import save_patterns_with_ids, save_patterns

def test_load_dataset_assigns_input_id(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("Glucose,BMI,Outcome\n120,25.0,0\n150,30.0,1\n")
    
    df, input_record = load_dataset_with_id(str(csv_file))
    assert input_record.input_id.startswith("INP-")
    assert input_record.row_count == 2
    assert input_record.column_count == 3
    assert "Glucose" in input_record.features
    assert df.shape == (2, 3)

def test_save_patterns_assigns_pattern_ids(tmp_path):
    raw_patterns = [
        {"type": "individual_dependency", "feature": "Glucose", "metrics": {"r": 0.46}, "description": "High Glucose"},
        {"type": "feature_combination_synergy", "feature": "Glucose + BMI", "metrics": {"gain": 0.05}, "description": "Synergy"}
    ]
    
    enriched = save_patterns_with_ids(raw_patterns, input_id="INP-20261104-TEST", output_dir=str(tmp_path))
    assert len(enriched) == 2
    assert enriched[0]["pattern_id"] == "PAT-001"
    assert enriched[0]["input_id"] == "INP-20261104-TEST"
    assert enriched[1]["pattern_id"] == "PAT-002"
    assert enriched[1]["input_id"] == "INP-20261104-TEST"
