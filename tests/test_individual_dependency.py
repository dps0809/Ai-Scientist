import pytest
import pandas as pd
import numpy as np
from modules.individual_dependency_analysis import individual_dependency_analysis

def test_individual_dependency_analysis_basic():
    # Synthetic dataset
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # y strongly correlated with x1, independent of x2
    y = (x1 * 2 + np.random.normal(0, 0.5, n) > 0).astype(int)

    df = pd.DataFrame({"feat1": x1, "feat2": x2, "target": y})

    results = individual_dependency_analysis(df, target="target")

    assert "ranked_features" in results
    assert "table" in results
    assert "patterns" in results
    assert len(results["table"]) == 2

    # feat1 should be ranked higher than feat2
    assert results["ranked_features"][0] == "feat1"
    feat1_data = next(r for r in results["table"] if r["feature"] == "feat1")
    assert feat1_data["mutual_information"] > 0
    assert feat1_data["pearson_r"] > 0
