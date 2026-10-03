import numpy as np
from typing import cast
from scipy.stats import pearsonr

def numerical_patterns(df, target):
    print("\n" + "=" * 70)
    print("5. NUMERICAL RELATIONSHIP DISCOVERY")
    print("=" * 70)

    patterns = []
    numerical = df.select_dtypes(include=np.number).columns.tolist()

    if target not in numerical:
        print("Target is not numerical. Skipping Pearson correlation.")
        return patterns

    for feature in numerical:
        if feature == target:
            continue
        x = df[feature]
        y = df[target]
        if x.nunique() < 2 or y.nunique() < 2:
            continue

        try:
            correlation_result = cast(tuple[float, float], pearsonr(x, y))
            correlation = float(correlation_result[0])
            p_value = float(correlation_result[1])

            if abs(correlation) >= 0.2 and p_value < 0.05:
                direction = "positive" if correlation > 0 else "negative"
                # Compute Fisher Z-score for Pearson correlation
                N = len(df)
                fisher_z = np.arctanh(correlation)
                z_score = round(float(fisher_z * np.sqrt(N - 3)), 2)
                p_val_str = "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"

                pattern = {
                    "type": "correlation",
                    "feature": str(feature),
                    "target": str(target),
                    "correlation": round(float(correlation), 4),
                    "z_score": z_score,
                    "p_value": p_val_str,
                    "direction": direction
                }
                patterns.append(pattern)
                print(f"{feature} -> {target} | r={correlation:.3f} | Z={z_score:+.2f} | p={p_val_str}")
        except Exception:
            pass

    return patterns
