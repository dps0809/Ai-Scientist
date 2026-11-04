import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def feature_importance(df, target, output_dir=None, force_recompute: bool = False):
    out_dir = output_dir or OUTPUT_DIR
    csv_path = os.path.join(out_dir, "feature_importance.csv")

    if os.path.exists(csv_path) and not force_recompute:
        try:
            cached_imp = pd.read_csv(csv_path, index_col=0).squeeze()
            patterns = []
            for feature, importance in cached_imp.items():
                if float(importance) >= 0.05:
                    patterns.append({
                        "type": "ml_feature_importance",
                        "feature": str(feature),
                        "target": str(target),
                        "importance": round(float(importance), 4),
                        "correlation": None,
                        "p_value": None,
                        "direction": None
                    })
            print(f"[CACHE HIT] Loaded feature importance patterns from: {csv_path}")
            return patterns
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("7. MACHINE LEARNING FEATURE IMPORTANCE")
    print("=" * 70)

    patterns = []
    try:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        for col in X.select_dtypes(include=["object", "category", "bool"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X = X.fillna(0)

        if y.dtype == "object" or str(y.dtype) == "category":
            y = LabelEncoder().fit_transform(y.astype(str))
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop features:")
        for feature, importance in importances.head(10).items():
            print(f"{feature:<30} {importance:.4f}")
            if importance >= 0.05:
                patterns.append({
                    "type": "ml_feature_importance",
                    "feature": str(feature),
                    "target": str(target),
                    "importance": round(float(importance), 4),
                    "correlation": None,
                    "p_value": None,
                    "direction": None
                })

        out_dir = output_dir or OUTPUT_DIR
        plots_dir = os.path.join(out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        importances.to_csv(os.path.join(out_dir, "feature_importance.csv"))

        plt.figure(figsize=(10, 6))
        importances.head(10).sort_values().plot(kind="barh")
        plt.title("Top Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
        plt.close()

    except Exception as e:
        print("Feature importance could not be calculated:")
        print(e)

    return patterns
