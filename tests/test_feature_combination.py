import pytest
import pandas as pd
import numpy as np
from modules.feature_combination_analysis import feature_combination_analysis

def test_feature_combination_analysis_synergy(tmp_path):
    np.random.seed(42)
    n = 150
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    # Target depends on both x1 and x2 synergistically
    y = ((x1 + x2 + np.random.normal(0, 0.5, n)) > 0).astype(int)

    df = pd.DataFrame({"feat1": x1, "feat2": x2, "feat3": x3, "target": y})

    results = feature_combination_analysis(
        df=df,
        target="target",
        top_candidates=["feat1", "feat2", "feat3"],
        max_candidates=3,
        cv_folds=3,
        output_dir=str(tmp_path)
    )

    assert "combinations" in results
    assert len(results["combinations"]) == 3  # (3 choose 2) = 3 pairs
    assert "top_synergies" in results

    pair_1_2 = next(r for r in results["combinations"] if "feat1" in r["combination"] and "feat2" in r["combination"])
    assert pair_1_2["combined_score"] >= pair_1_2["score_a"]
    assert pair_1_2["joint_mi"] > 0
