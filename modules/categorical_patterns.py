import numpy as np
import pandas as pd
from typing import cast
from scipy.stats import ttest_ind, chi2_contingency

def categorical_patterns(df, target):
    print("\n" + "=" * 70)
    print("6. CATEGORICAL RELATIONSHIP DISCOVERY")
    print("=" * 70)

    patterns = []
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = df.select_dtypes(include=np.number).columns.tolist()

    if target in numerical:
        for feature in categorical:
            if df[feature].nunique() > 15:
                continue
            groups = [group[target].dropna().values for _, group in df.groupby(feature)]
            if len(groups) != 2:
                continue

            try:
                test_result = cast(tuple[float, float], ttest_ind(groups[0], groups[1], equal_var=False))
                p_value = float(test_result[1])
                means = df.groupby(feature)[target].mean()

                if p_value < 0.05:
                    pattern = {
                        "type": "group_difference",
                        "feature": str(feature),
                        "target": str(target),
                        "group_means": {str(k): round(float(v), 4) for k, v in means.items()},
                        "p_value": round(float(p_value), 6)
                    }
                    patterns.append(pattern)
                    print(f"{feature} differs significantly across groups | p={p_value:.5f}")
            except Exception:
                pass

    if target in categorical:
        for feature in categorical:
            if feature == target:
                continue
            if df[feature].nunique() > 20:
                continue
            try:
                table = pd.crosstab(df[feature], df[target])
                chi_square_result = cast(tuple[float, float, object, object], chi2_contingency(table))
                chi2 = float(chi_square_result[0])
                p_value = float(chi_square_result[1])

                if p_value < 0.05:
                    pattern = {
                        "type": "categorical_association",
                        "feature": str(feature),
                        "target": str(target),
                        "chi_square": round(float(chi2), 4),
                        "p_value": round(float(p_value), 6)
                    }
                    patterns.append(pattern)
                    print(f"{feature} <-> {target} | chi^2={chi2:.3f} | p={p_value:.5f}")
            except Exception:
                pass

    return patterns
