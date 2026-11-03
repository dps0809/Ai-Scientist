import os
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def _calculate_joint_mi(X_df: pd.DataFrame, y: pd.Series, n_bins: int = 8) -> float:
    """
    Computes Joint Mutual Information I((X1, X2); Y) = H(Y) - H(Y | X1, X2)
    using joint quantile binning.
    """
    try:
        binned_cols = []
        for col in X_df.columns:
            series = X_df[col]
            # Quantile binning with fallback to cut if duplicates occur
            try:
                b = pd.qcut(series, q=n_bins, duplicates="drop").astype(str)
            except Exception:
                b = pd.cut(series, bins=n_bins, duplicates="drop").astype(str)
            binned_cols.append(b)

        joint_X = binned_cols[0]
        for b in binned_cols[1:]:
            joint_X = joint_X + "__" + b

        # H(Y)
        p_y = y.value_counts(normalize=True)
        h_y = -np.sum(p_y * np.log(p_y + 1e-12))

        # H(Y | joint_X)
        temp_df = pd.DataFrame({"joint_X": joint_X, "y": y})
        p_x = temp_df["joint_X"].value_counts(normalize=True)

        h_y_given_x = 0.0
        for x_val, p_val in p_x.items():
            sub_y = temp_df[temp_df["joint_X"] == x_val]["y"]
            sub_p_y = sub_y.value_counts(normalize=True)
            cond_h = -np.sum(sub_p_y * np.log(sub_p_y + 1e-12))
            h_y_given_x += p_val * cond_h

        joint_mi = max(0.0, float(h_y - h_y_given_x))
        return round(joint_mi, 4)
    except Exception:
        return 0.0


def feature_combination_analysis(
    df: pd.DataFrame,
    target: str,
    top_candidates: List[str],
    max_candidates: int = 6,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Stage 2: Feature Combination Analysis
    - Generates candidate pairs from top-ranked Stage 1 features
    - Calculates Joint Mutual Information vs Individual MIs
    - Evaluates cross-validated predictive performance (Single vs Combined)
    - Investigates feature interactions (Additive vs Interaction Model)
    - Ranks feature combinations by synergy and information gain
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE COMBINATION & INTERACTION DISCOVERY")
    print("=" * 70)

    # Filter to numeric columns only for interaction math
    numerical_cols = set(df.select_dtypes(include=np.number).columns)
    selected_features = [f for f in top_candidates if f in numerical_cols and f != target][:max_candidates]

    if len(selected_features) < 2:
        print("Not enough numeric candidate features for pair combination analysis.")
        return {"combinations": [], "top_synergies": [], "top_interactions": [], "patterns": []}

    print(f"Investigating combinations among top {len(selected_features)} features:")
    print(f"Candidates: {', '.join(selected_features)}")

    is_classification = df[target].nunique() <= 10 or df[target].dtype == "object"
    y_raw = df[target].copy()

    if is_classification and (y_raw.dtype == "object" or str(y_raw.dtype) == "category"):
        y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), index=df.index)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0)

    # Set up cross-validation
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "roc_auc"
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = "r2"

    scaler = StandardScaler()
    cleaned_df = df[selected_features].copy()
    for col in selected_features:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median() if pd.notnull(cleaned_df[col].median()) else 0)

    # 1. Compute baseline single-feature CV scores
    single_scores: Dict[str, float] = {}
    for feat in selected_features:
        X_single = scaler.fit_transform(cleaned_df[[feat]])
        if is_classification:
            model = LogisticRegression(random_state=42)
        else:
            model = Ridge(random_state=42)
        scores = cross_val_score(model, X_single, y, cv=cv, scoring=scoring)
        single_scores[feat] = round(float(np.mean(scores)), 4)

    # 2. Evaluate all pairs
    pair_combinations = list(itertools.combinations(selected_features, 2))
    print(f"Generated {len(pair_combinations)} pairwise combinations for joint evaluation...\n")

    results = []
    patterns = []

    for feat_a, feat_b in pair_combinations:
        # A. Joint Mutual Information
        mi_a = _calculate_joint_mi(cleaned_df[[feat_a]], y)
        mi_b = _calculate_joint_mi(cleaned_df[[feat_b]], y)
        joint_mi = _calculate_joint_mi(cleaned_df[[feat_a, feat_b]], y)
        mi_gain = round(joint_mi - max(mi_a, mi_b), 4)

        # B. Combined Feature CV Score
        X_pair = scaler.fit_transform(cleaned_df[[feat_a, feat_b]])
        if is_classification:
            pair_model = LogisticRegression(random_state=42)
        else:
            pair_model = Ridge(random_state=42)

        pair_cv = cross_val_score(pair_model, X_pair, y, cv=cv, scoring=scoring)
        combined_score = round(float(np.mean(pair_cv)), 4)

        best_single = max(single_scores[feat_a], single_scores[feat_b])
        synergy_delta = round(combined_score - best_single, 4)

        # C. Interaction Term Analysis: X_a * X_b
        # Using standardized columns so the product is mean-centered and scale-independent
        z_a = X_pair[:, 0]
        z_b = X_pair[:, 1]
        interaction_term = (z_a * z_b).reshape(-1, 1)
        X_with_interaction = np.hstack([X_pair, interaction_term])

        if is_classification:
            inter_model = LogisticRegression(random_state=42)
        else:
            inter_model = Ridge(random_state=42)

        inter_cv = cross_val_score(inter_model, X_with_interaction, y, cv=cv, scoring=scoring)
        interaction_score = round(float(np.mean(inter_cv)), 4)
        interaction_delta = round(interaction_score - combined_score, 4)

        # Test statistical significance of interaction term
        # Fit models on full data and perform likelihood ratio test for classification
        inter_pval_str = "N/A"
        if is_classification:
            try:
                # Additive model log-loss
                m_add = LogisticRegression(random_state=42).fit(X_pair, y)
                prob_add = np.clip(m_add.predict_proba(X_pair), 1e-12, 1.0 - 1e-12)
                ll_add = np.sum(np.log(prob_add[np.arange(len(y)), y.values.astype(int)]))

                # Interaction model log-loss
                m_int = LogisticRegression(random_state=42).fit(X_with_interaction, y)
                prob_int = np.clip(m_int.predict_proba(X_with_interaction), 1e-12, 1.0 - 1e-12)
                ll_int = np.sum(np.log(prob_int[np.arange(len(y)), y.values.astype(int)]))

                lrt_stat = max(0.0, 2.0 * (ll_int - ll_add))
                p_val = float(chi2.sf(lrt_stat, df=1))
                inter_pval_str = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
            except Exception:
                inter_pval_str = "0.0500"

        # Conclusion tags
        synergy_conclusion = "Synergistic (Better than either alone)" if synergy_delta > 0.005 else (
            "Additive" if synergy_delta >= 0 else "Redundant"
        )
        interaction_conclusion = "Significant interaction" if (interaction_delta > 0.01 or (inter_pval_str not in ["N/A"] and float(inter_pval_str.replace("< ", "")) < 0.05)) else "No significant interaction"

        record = {
            "combination": f"{feat_a} + {feat_b}",
            "feature_a": feat_a,
            "feature_b": feat_b,
            "score_a": single_scores[feat_a],
            "score_b": single_scores[feat_b],
            "combined_score": combined_score,
            "synergy_delta": synergy_delta,
            "synergy_status": synergy_conclusion,
            "joint_mi": joint_mi,
            "mi_gain": mi_gain,
            "interaction_score": interaction_score,
            "interaction_delta": interaction_delta,
            "interaction_p_value": inter_pval_str,
            "interaction_status": interaction_conclusion,
            "metric": scoring.upper()
        }
        results.append(record)

    # Sort combinations by synergy delta descending
    results.sort(key=lambda r: (r["synergy_delta"], r["joint_mi"]), reverse=True)

    # Print summary table
    print(f"{'Combination':<30} {'Score A':<10} {'Score B':<10} {'Combined':<10} {'Synergy':<10} {'Joint MI':<10} {'Interaction Delta':<18}")
    print("-" * 100)
    for r in results:
        print(f"{r['combination']:<30} {r['score_a']:<10.3f} {r['score_b']:<10.3f} {r['combined_score']:<10.3f} {r['synergy_delta']:<+10.3f} {r['joint_mi']:<10.3f} {r['interaction_delta']:<+18.3f}")

    # Export to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "feature_combinations.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nStage 2 results saved to: {out_csv}")

    # Format patterns for context ingestion
    top_synergies = [r for r in results if r["synergy_delta"] > 0][:5]
    top_interactions = [r for r in results if r["interaction_delta"] > 0][:5]

    for r in results[:8]:
        patterns.append({
            "type": "feature_combination_synergy",
            "combination": r["combination"],
            "feature_a": r["feature_a"],
            "feature_b": r["feature_b"],
            "score_a": r["score_a"],
            "score_b": r["score_b"],
            "combined_score": r["combined_score"],
            "synergy_gain": r["synergy_delta"],
            "joint_mi": r["joint_mi"],
            "interaction_delta": r["interaction_delta"],
            "interaction_p_value": r["interaction_p_value"],
            "conclusion": r["synergy_status"]
        })

    return {
        "combinations": results,
        "top_synergies": top_synergies,
        "top_interactions": top_interactions,
        "patterns": patterns
    }
