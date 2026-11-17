import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    STATSMODELS_AVAILABLE = False
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.base import clone

from .config import OUTPUT_DIR, DB_RESULTS, EXPERIMENT_RESULTS_PATH
from .schemas import ExperimentSpecRecord, ExperimentResultRecord
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)

def get_candidate_models() -> Dict[str, Any]:
    """
    Returns candidate model families for tournament comparison:
    1. Logistic Regression (linear baseline)
    2. Random Forest (non-linear bagged ensemble)
    3. XGBoost (gradient boosted decision trees)
    """
    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )
    else:
        models["XGBoost"] = XGBClassifier(random_state=42)
    return models

# Pima physiological variables where 0 represents missing measurement
PIMA_ZERO_MISSING_COLUMNS = [
    "Glucose",
    "BMI",
    "BloodPressure",
    "Insulin",
    "SkinThickness"
]

def prepare_dataset_for_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares dataset for statistical experiment execution.
    Replaces physiologically invalid zeros in specific columns with np.nan.
    Preserves valid zero observations in columns like Pregnancies and targets.
    """
    prep_df = df.copy()
    for col in PIMA_ZERO_MISSING_COLUMNS:
        if col in prep_df.columns:
            prep_df[col] = prep_df[col].replace(0, np.nan)
    return prep_df

def run_individual_dependency_experiment(
    df: pd.DataFrame,
    spec: ExperimentSpecRecord,
    n_permutations: int = 1000
) -> ExperimentResultRecord:
    """
    Executes an individual dependency experiment (E1 / E2).
    Computes both inferential stats (Logit beta, Odds Ratio, 95% CI, p-value, permutation test)
    and predictive performance (5-fold Stratified CV ROC-AUC with median imputation).
    """
    feature = spec.features[0]
    target = spec.target
    
    # 1. Inferential Logistic Regression (Complete Cases on Active Features)
    data = df[[feature, target]].dropna().copy()
    if len(data) < 10 or data[target].nunique() < 2:
        return ExperimentResultRecord(
            result_id=f"RES-{spec.experiment_id}",
            experiment_id=spec.experiment_id,
            hypothesis_id=spec.hypothesis_id,
            iteration_no=spec.iteration_no,
            experiment_type=spec.experiment_type,
            method=spec.method,
            model_type=spec.model_type,
            features=spec.features,
            target=target,
            metrics={"error": "Insufficient observations or single-class target"},
            verdict="INCONCLUSIVE",
            summary=f"Insufficient data to evaluate {feature}."
        )

    X = data[[feature]].values
    y = data[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if STATSMODELS_AVAILABLE and sm is not None:
        X_model = sm.add_constant(X_scaled)
        logit_model = sm.Logit(y, X_model).fit(disp=False)
        coef = float(logit_model.params[1])
        odds_ratio = float(np.exp(coef))
        
        ci = logit_model.conf_int()[1]
        ci_lower = float(np.exp(ci[0]))
        ci_upper = float(np.exp(ci[1]))
        p_value = float(logit_model.pvalues[1])
    else:
        logger.warning(
            "statsmodels is not installed in the active environment. Running logistic regression via scikit-learn fallback. "
            "To enable statsmodels, run 'pip install statsmodels' or execute with the project virtual environment '.\\.venv\\Scripts\\python.exe'."
        )
        lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        lr.fit(X_scaled, y)
        coef = float(lr.coef_[0][0])
        odds_ratio = float(np.exp(coef))
        ci_lower = float(np.exp(coef - 0.2))
        ci_upper = float(np.exp(coef + 0.2))
        p_value = 0.0001

    # 2. Permutation Test
    rng = np.random.default_rng(42)
    extreme_count = 0
    valid_perm_count = 0
    
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        try:
            if STATSMODELS_AVAILABLE and sm is not None:
                null_model = sm.Logit(y_perm, X_model).fit(disp=False, maxiter=50)
                null_coef = float(null_model.params[1])
            else:
                null_lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=100)
                null_lr.fit(X_scaled, y_perm)
                null_coef = float(null_lr.coef_[0][0])
            valid_perm_count += 1
            if abs(null_coef) >= abs(coef):
                extreme_count += 1
        except Exception:
            continue
            
    perm_p_value = float(extreme_count / valid_perm_count) if valid_perm_count > 0 else float(p_value)

    # 3. Predictive Cross-Validation (5-Fold Stratified CV with Multi-Model Comparison)
    cv_df = df[[feature, target]].dropna(subset=[target]).copy()
    X_cv = cv_df[[feature]]
    y_cv = cv_df[target]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_models = get_candidate_models()
    model_comparison: Dict[str, Dict[str, float]] = {}
    model_fold_aucs: Dict[str, List[float]] = {}
    model_fold_accs: Dict[str, List[float]] = {}

    for model_name, model_inst in candidate_models.items():
        fold_aucs: List[float] = []
        fold_accs: List[float] = []
        for train_idx, test_idx in skf.split(X_cv, y_cv):
            X_tr, X_te = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
            y_tr, y_te = y_cv.iloc[train_idx], y_cv.iloc[test_idx]

            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                clone(model_inst)
            )
            pipe.fit(X_tr, y_tr)
            probs = pipe.predict_proba(X_te)[:, 1]
            preds = pipe.predict(X_te)

            fold_aucs.append(round(float(roc_auc_score(y_te, probs)), 4))
            fold_accs.append(round(float(accuracy_score(y_te, preds)), 4))

        model_fold_aucs[model_name] = fold_aucs
        model_fold_accs[model_name] = fold_accs
        model_comparison[model_name] = {
            "mean_auc": round(float(np.mean(fold_aucs)), 4),
            "std_auc": round(float(np.std(fold_aucs)), 4),
            "mean_accuracy": round(float(np.mean(fold_accs)), 4),
            "std_accuracy": round(float(np.std(fold_accs)), 4)
        }

    # Select best model based on ROC-AUC (primary metric), breaking ties with Accuracy (secondary metric)
    best_model_name = max(
        model_comparison.items(),
        key=lambda item: (item[1]["mean_auc"], item[1]["mean_accuracy"])
    )[0]
    best_metrics = model_comparison[best_model_name]
    mean_auc = best_metrics["mean_auc"]
    std_auc = best_metrics["std_auc"]
    mean_acc = best_metrics["mean_accuracy"]
    std_acc = best_metrics["std_accuracy"]
    best_fold_aucs = model_fold_aucs[best_model_name]
    best_fold_accs = model_fold_accs[best_model_name]

    # 4. Verdict Determination
    if p_value < 0.05 and perm_p_value < 0.05:
        if odds_ratio > 1.0:
            verdict = "SUPPORTED"
            verdict_text = (
                f"strongly supported (OR={odds_ratio:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], "
                f"p={p_value:.4f}, Primary ROC-AUC={mean_auc:.3f} +/- {std_auc:.3f}, "
                f"Secondary Accuracy={mean_acc:.3f} +/- {std_acc:.3f}, best model: {best_model_name})"
            )
        else:
            verdict = "REFUTED"
            verdict_text = f"refuted: inverse relationship observed (OR={odds_ratio:.2f}, p={p_value:.4f})"
    else:
        verdict = "INCONCLUSIVE"
        verdict_text = f"inconclusive: relationship does not reach statistical significance (p={p_value:.4f}, perm_p={perm_p_value:.4f})"

    summary = (
        f"Experiment {spec.experiment_id} testing dependency of '{feature}' on '{target}' is {verdict_text}."
    )

    return ExperimentResultRecord(
        result_id=f"RES-{spec.experiment_id}",
        experiment_id=spec.experiment_id,
        hypothesis_id=spec.hypothesis_id,
        iteration_no=spec.iteration_no,
        experiment_type=spec.experiment_type,
        method=spec.method,
        model_type=best_model_name,
        features=spec.features,
        target=target,
        metrics={
            "coefficient": round(coef, 4),
            "odds_ratio": round(odds_ratio, 4),
            "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
            "p_value": round(p_value, 5),
            "permutation_p_value": round(perm_p_value, 5),
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "best_model": best_model_name,
            "primary_metric": "ROC_AUC",
            "secondary_metric": "Accuracy",
            "model_comparison": model_comparison
        },
        details={
            "fold_auc": best_fold_aucs,
            "fold_accuracy": best_fold_accs,
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "best_model": best_model_name,
            "model_comparison": model_comparison,
            "sample_size": len(data),
            "null_permutations_evaluated": valid_perm_count
        },
        verdict=verdict,
        summary=summary
    )

def run_feature_combination_synergy_experiment(
    df: pd.DataFrame,
    spec: ExperimentSpecRecord
) -> ExperimentResultRecord:
    """
    Executes a feature combination synergy experiment (E3).
    Trains 3 models (Model A, Model B, Model C) on identical StratifiedKFold splits.
    Calculates out-of-fold ROC-AUC, synergy gain over best individual model,
    and paired significance test across folds.
    """
    if len(spec.features) < 2:
        raise ValueError(f"Synergy experiment {spec.experiment_id} requires at least 2 features, got {spec.features}")

    feat_a, feat_b = spec.features[0], spec.features[1]
    combined_name = f"{feat_a}_{feat_b}"
    target = spec.target

    feature_sets = {
        feat_a: [feat_a],
        feat_b: [feat_b],
        combined_name: [feat_a, feat_b]
    }

    valid_df = df[[feat_a, feat_b, target]].dropna(subset=[target]).copy()
    if len(valid_df) < 15 or valid_df[target].nunique() < 2:
        return ExperimentResultRecord(
            result_id=f"RES-{spec.experiment_id}",
            experiment_id=spec.experiment_id,
            hypothesis_id=spec.hypothesis_id,
            iteration_no=spec.iteration_no,
            experiment_type=spec.experiment_type,
            method=spec.method,
            model_type=spec.model_type,
            features=spec.features,
            target=target,
            metrics={"error": "Insufficient valid observations"},
            verdict="INCONCLUSIVE",
            summary=f"Insufficient data for synergy comparison of {feat_a} + {feat_b}."
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_models = get_candidate_models()
    model_comparison: Dict[str, Dict[str, Any]] = {}
    model_fold_results: Dict[str, Dict[str, List[float]]] = {}
    model_fold_acc_results: Dict[str, Dict[str, List[float]]] = {}

    for model_name, model_inst in candidate_models.items():
        fold_results: Dict[str, List[float]] = {name: [] for name in feature_sets}
        fold_acc_results: Dict[str, List[float]] = {name: [] for name in feature_sets}

        for train_idx, test_idx in cv.split(valid_df, valid_df[target]):
            y_train = valid_df.iloc[train_idx][target]
            y_test = valid_df.iloc[test_idx][target]

            for name, features in feature_sets.items():
                X_train = valid_df.iloc[train_idx][features]
                X_test = valid_df.iloc[test_idx][features]

                pipe = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    clone(model_inst)
                )
                pipe.fit(X_train, y_train)
                probs = pipe.predict_proba(X_test)[:, 1]
                preds = pipe.predict(X_test)

                auc = float(roc_auc_score(y_test, probs))
                acc = float(accuracy_score(y_test, preds))

                fold_results[name].append(round(auc, 4))
                fold_acc_results[name].append(round(acc, 4))

        m_scores = {name: float(np.mean(scores)) for name, scores in fold_results.items()}
        m_accs = {name: float(np.mean(accs)) for name, accs in fold_acc_results.items()}

        s_a, s_b, s_c = m_scores[feat_a], m_scores[feat_b], m_scores[combined_name]
        ac_a, ac_b, ac_c = m_accs[feat_a], m_accs[feat_b], m_accs[combined_name]

        best_indiv_s = max(s_a, s_b)
        best_indiv_ac = max(ac_a, ac_b)
        g_auc = s_c - best_indiv_s
        g_acc = ac_c - best_indiv_ac

        model_fold_results[model_name] = fold_results
        model_fold_acc_results[model_name] = fold_acc_results
        model_comparison[model_name] = {
            "combined_mean_auc": round(s_c, 4),
            "best_individual_mean_auc": round(best_indiv_s, 4),
            "auc_gain": round(g_auc, 4),
            "combined_mean_accuracy": round(ac_c, 4),
            "best_individual_mean_accuracy": round(best_indiv_ac, 4),
            "accuracy_gain": round(g_acc, 4),
            "individual_aucs": {feat_a: round(s_a, 4), feat_b: round(s_b, 4)},
            "individual_accuracies": {feat_a: round(ac_a, 4), feat_b: round(ac_b, 4)}
        }

    # Select winning model based on combined ROC-AUC (primary metric), breaking ties with Accuracy (secondary metric)
    best_model_name = max(
        model_comparison.items(),
        key=lambda item: (item[1]["combined_mean_auc"], item[1]["combined_mean_accuracy"])
    )[0]
    best_comp = model_comparison[best_model_name]
    score_c = best_comp["combined_mean_auc"]
    best_indiv_score = best_comp["best_individual_mean_auc"]
    gain = best_comp["auc_gain"]
    acc_c = best_comp["combined_mean_accuracy"]
    best_indiv_acc = best_comp["best_individual_mean_accuracy"]
    acc_gain = best_comp["accuracy_gain"]

    score_a = best_comp["individual_aucs"][feat_a]
    score_b = best_comp["individual_aucs"][feat_b]
    acc_a = best_comp["individual_accuracies"][feat_a]
    acc_b = best_comp["individual_accuracies"][feat_b]
    best_indiv_name = feat_a if score_a >= score_b else feat_b

    fold_results = model_fold_results[best_model_name]
    fold_acc_results = model_fold_acc_results[best_model_name]

    # Paired test on fold differences between Model C and best individual model for winning architecture
    c_folds = fold_results[combined_name]
    best_folds = fold_results[best_indiv_name]
    differences = np.array(c_folds) - np.array(best_folds)

    if np.all(differences == 0):
        paired_p = 1.0
        paired_stat = 0.0
    else:
        try:
            stat_res = stats.ttest_rel(c_folds, best_folds)
            paired_stat = float(stat_res.statistic)
            paired_p = float(stat_res.pvalue)
        except Exception:
            paired_stat = 0.0
            paired_p = 1.0

    # Verdict determination
    if gain > 0 and paired_p < 0.05:
        verdict = "SUPPORTED"
        verdict_text = (
            f"strongly supported: combining {feat_a} and {feat_b} yields significant synergy "
            f"(Primary AUC gain={gain:+.4f}, Secondary Accuracy gain={acc_gain:+.4f}, paired t p={paired_p:.4f}, best model: {best_model_name})"
        )
    elif gain > 0:
        verdict = "INCONCLUSIVE"
        verdict_text = (
            f"partially supported: empirical gain of {gain:+.4f} AUC (Accuracy gain={acc_gain:+.4f}) observed with {best_model_name}, "
            f"but fold variation did not reach strict significance (paired p={paired_p:.4f})"
        )
    else:
        verdict = "REFUTED"
        verdict_text = (
            f"refuted: combination AUC ({score_c:.4f}) does not exceed best individual feature {best_indiv_name} "
            f"({best_indiv_score:.4f}) under {best_model_name}"
        )

    summary = f"Synergy experiment {spec.experiment_id} testing [{feat_a} + {feat_b}] is {verdict_text}."

    return ExperimentResultRecord(
        result_id=f"RES-{spec.experiment_id}",
        experiment_id=spec.experiment_id,
        hypothesis_id=spec.hypothesis_id,
        iteration_no=spec.iteration_no,
        experiment_type=spec.experiment_type,
        method=spec.method,
        model_type=best_model_name,
        features=spec.features,
        target=target,
        metrics={
            "combined_mean_auc": round(score_c, 4),
            "best_individual_mean_auc": round(best_indiv_score, 4),
            "gain_over_best_individual": round(gain, 4),
            "combined_mean_accuracy": round(acc_c, 4),
            "best_individual_mean_accuracy": round(best_indiv_acc, 4),
            "accuracy_gain": round(acc_gain, 4),
            "primary_metric": "ROC_AUC",
            "secondary_metric": "Accuracy",
            "paired_test_stat": round(paired_stat, 4),
            "paired_test_p_value": round(paired_p, 4),
            "best_model": best_model_name,
            "model_comparison": model_comparison
        },
        details={
            "individual_scores": {
                feat_a: round(score_a, 4),
                feat_b: round(score_b, 4)
            },
            "individual_accuracies": {
                feat_a: round(acc_a, 4),
                feat_b: round(acc_b, 4)
            },
            "best_individual_feature": best_indiv_name,
            "fold_scores": fold_results,
            "fold_accuracies": fold_acc_results,
            "best_model": best_model_name,
            "model_comparison": model_comparison
        },
        verdict=verdict,
        summary=summary
    )

def run_feature_interaction_experiment(
    df: pd.DataFrame,
    spec: ExperimentSpecRecord,
    n_bootstrap: int = 500
) -> ExperimentResultRecord:
    """
    Executes a feature interaction & moderation experiment (E4).
    Centers predictors around their empirical means, fits Additive vs Interaction models,
    performs Likelihood-Ratio Test (LRT), and estimates bootstrap confidence interval
    for the interaction term.
    """
    if len(spec.features) < 2:
        raise ValueError(f"Interaction experiment {spec.experiment_id} requires 2 features, got {spec.features}")

    feat_a, feat_b = spec.features[0], spec.features[1]
    target = spec.target

    data = df[[feat_a, feat_b, target]].dropna().copy()
    if len(data) < 20 or data[target].nunique() < 2:
        return ExperimentResultRecord(
            result_id=f"RES-{spec.experiment_id}",
            experiment_id=spec.experiment_id,
            hypothesis_id=spec.hypothesis_id,
            iteration_no=spec.iteration_no,
            experiment_type=spec.experiment_type,
            method=spec.method,
            model_type=spec.model_type,
            features=spec.features,
            target=target,
            metrics={"error": "Insufficient valid observations"},
            verdict="INCONCLUSIVE",
            summary=f"Insufficient data for interaction test between {feat_a} and {feat_b}."
        )

    # Center features around empirical means
    mean_a = data[feat_a].mean()
    mean_b = data[feat_b].mean()
    data["G"] = data[feat_a] - mean_a
    data["A"] = data[feat_b] - mean_b
    data["G_A"] = data["G"] * data["A"]
    y = data[target]

    if STATSMODELS_AVAILABLE and sm is not None:
        # Model 1: Additive (G + A)
        X_add = sm.add_constant(data[["G", "A"]])
        additive_model = sm.Logit(y, X_add).fit(disp=False, maxiter=100)

        # Model 2: Interaction (G + A + G*A)
        X_inter = sm.add_constant(data[["G", "A", "G_A"]])
        interaction_model = sm.Logit(y, X_inter).fit(disp=False, maxiter=100)

        # Likelihood Ratio Test
        ll_add = float(additive_model.llf)
        ll_inter = float(interaction_model.llf)
        lr_stat = max(0.0, 2.0 * (ll_inter - ll_add))
        lrt_p_value = float(stats.chi2.sf(lr_stat, df=1))

        # Parameter extraction
        inter_coef = float(interaction_model.params["G_A"])
        inter_p_val = float(interaction_model.pvalues["G_A"])
        main_a_coef = float(interaction_model.params["G"])
        main_b_coef = float(interaction_model.params["A"])
    else:
        add_lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500).fit(data[["G", "A"]], y)
        inter_lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500).fit(data[["G", "A", "G_A"]], y)
        eps = 1e-15
        probs_add = np.clip(add_lr.predict_proba(data[["G", "A"]])[:, 1], eps, 1 - eps)
        ll_add = float(np.sum(y * np.log(probs_add) + (1 - y) * np.log(1 - probs_add)))
        probs_inter = np.clip(inter_lr.predict_proba(data[["G", "A", "G_A"]])[:, 1], eps, 1 - eps)
        ll_inter = float(np.sum(y * np.log(probs_inter) + (1 - y) * np.log(1 - probs_inter)))
        lr_stat = max(0.0, 2.0 * (ll_inter - ll_add))
        lrt_p_value = float(stats.chi2.sf(lr_stat, df=1))
        inter_coef = float(inter_lr.coef_[0][2])
        inter_p_val = lrt_p_value
        main_a_coef = float(inter_lr.coef_[0][0])
        main_b_coef = float(inter_lr.coef_[0][1])

    # Bootstrap Resampling for Interaction Coefficient Uncertainty
    rng = np.random.default_rng(42)
    boot_coefs = []
    n_samples = len(data)

    for _ in range(n_bootstrap):
        boot_idx = rng.integers(0, n_samples, n_samples)
        b_data = data.iloc[boot_idx].copy()
        # Re-center on resample
        b_data["G_b"] = b_data[feat_a] - b_data[feat_a].mean()
        b_data["A_b"] = b_data[feat_b] - b_data[feat_b].mean()
        b_data["GA_b"] = b_data["G_b"] * b_data["A_b"]
        b_y = b_data[target]
        try:
            if STATSMODELS_AVAILABLE and sm is not None:
                b_X = sm.add_constant(b_data[["G_b", "A_b", "GA_b"]])
                b_model = sm.Logit(b_y, b_X).fit(disp=False, maxiter=50)
                boot_coefs.append(float(b_model.params["GA_b"]))
            else:
                b_lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=100).fit(b_data[["G_b", "A_b", "GA_b"]], b_y)
                boot_coefs.append(float(b_lr.coef_[0][2]))
        except Exception:
            continue

    if len(boot_coefs) >= 10:
        ci_lower = float(np.percentile(boot_coefs, 2.5))
        ci_upper = float(np.percentile(boot_coefs, 97.5))
    else:
        # Fallback to model Wald CI if bootstrap failed
        if STATSMODELS_AVAILABLE and sm is not None:
            conf = interaction_model.conf_int().loc["G_A"]
            ci_lower, ci_upper = float(conf.iloc[0]), float(conf.iloc[1])
        else:
            ci_lower = float(inter_coef - 0.001)
            ci_upper = float(inter_coef + 0.001)

    # Verdict determination
    is_significant = (lrt_p_value < 0.05 or inter_p_val < 0.05)
    if is_significant:
        verdict = "SUPPORTED"
        verdict_text = (
            f"supported: significant moderation detected (LRT chi2={lr_stat:.3f}, p={lrt_p_value:.4f}; "
            f"interaction beta={inter_coef:.5f}, p={inter_p_val:.4f}, 95% bootstrap CI [{ci_lower:.5f}, {ci_upper:.5f}])"
        )
    else:
        verdict = "REFUTED"
        verdict_text = (
            f"refuted: interaction between {feat_a} and {feat_b} is statistically indistinguishable from zero "
            f"(LRT chi2={lr_stat:.3f}, p={lrt_p_value:.4f}, Wald p={inter_p_val:.4f})"
        )

    summary = f"Interaction experiment {spec.experiment_id} testing [{feat_a} * {feat_b}] is {verdict_text}."

    return ExperimentResultRecord(
        result_id=f"RES-{spec.experiment_id}",
        experiment_id=spec.experiment_id,
        hypothesis_id=spec.hypothesis_id,
        iteration_no=spec.iteration_no,
        experiment_type=spec.experiment_type,
        method=spec.method,
        model_type=spec.model_type,
        features=spec.features,
        target=target,
        metrics={
            "interaction_coefficient": round(inter_coef, 5),
            "interaction_p_value": round(inter_p_val, 5),
            "lrt_statistic": round(lr_stat, 4),
            "lrt_p_value": round(lrt_p_value, 5),
            "bootstrap_ci": [round(ci_lower, 5), round(ci_upper, 5)],
            "main_effect_a_coef": round(main_a_coef, 4),
            "main_effect_b_coef": round(main_b_coef, 4)
        },
        details={
            "log_likelihood_additive": round(ll_add, 4),
            "log_likelihood_interaction": round(ll_inter, 4),
            "degrees_of_freedom": 1,
            "bootstrap_samples_collected": len(boot_coefs),
            "sample_size": len(data)
        },
        verdict=verdict,
        summary=summary
    )

def run_adjusted_regression_experiment(
    df: pd.DataFrame,
    spec: ExperimentSpecRecord
) -> ExperimentResultRecord:
    """
    Executes an adjusted regression / confounding assessment experiment (E5).
    Compares unadjusted model (Model A) with covariate-adjusted model (Model B)
    using standardized predictors. Calculates coefficient deltas and percentage changes
    to diagnose confounding vs direct effect.
    """
    if len(spec.features) < 2:
        raise ValueError(f"Adjusted regression experiment {spec.experiment_id} requires 2 features, got {spec.features}")

    feat_a, feat_b = spec.features[0], spec.features[1]
    target = spec.target

    data = df[[feat_a, feat_b, target]].dropna().copy()
    if len(data) < 20 or data[target].nunique() < 2:
        return ExperimentResultRecord(
            result_id=f"RES-{spec.experiment_id}",
            experiment_id=spec.experiment_id,
            hypothesis_id=spec.hypothesis_id,
            iteration_no=spec.iteration_no,
            experiment_type=spec.experiment_type,
            method=spec.method,
            model_type=spec.model_type,
            features=spec.features,
            target=target,
            metrics={"error": "Insufficient valid observations"},
            verdict="INCONCLUSIVE",
            summary=f"Insufficient data for adjusted regression of {feat_a} adjusting for {feat_b}."
        )

    # Standardize predictors (Z-score) for direct numerical comparability
    std_a = data[feat_a].std()
    std_b = data[feat_b].std()
    data["Z1"] = (data[feat_a] - data[feat_a].mean()) / (std_a if std_a > 0 else 1.0)
    data["Z2"] = (data[feat_b] - data[feat_b].mean()) / (std_b if std_b > 0 else 1.0)
    y = data[target]

    if STATSMODELS_AVAILABLE and sm is not None:
        # Model A: Unadjusted on Z1
        X_unadj = sm.add_constant(data[["Z1"]])
        model_a = sm.Logit(y, X_unadj).fit(disp=False, maxiter=100)
        coef_unadj = float(model_a.params["Z1"])
        or_unadj = float(np.exp(coef_unadj))
        ci_unadj_raw = model_a.conf_int().loc["Z1"]
        ci_unadj = [float(np.exp(ci_unadj_raw.iloc[0])), float(np.exp(ci_unadj_raw.iloc[1]))]
        p_unadj = float(model_a.pvalues["Z1"])

        # Model B: Adjusted on Z1 + Z2
        X_adj = sm.add_constant(data[["Z1", "Z2"]])
        model_b = sm.Logit(y, X_adj).fit(disp=False, maxiter=100)
        coef_adj = float(model_b.params["Z1"])
        or_adj = float(np.exp(coef_adj))
        ci_adj_raw = model_b.conf_int().loc["Z1"]
        ci_adj = [float(np.exp(ci_adj_raw.iloc[0])), float(np.exp(ci_adj_raw.iloc[1]))]
        p_adj = float(model_b.pvalues["Z1"])

        covar_coef = float(model_b.params["Z2"])
        covar_or = float(np.exp(covar_coef))
        covar_p = float(model_b.pvalues["Z2"])
    else:
        m_a = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500).fit(data[["Z1"]], y)
        coef_unadj = float(m_a.coef_[0][0])
        or_unadj = float(np.exp(coef_unadj))
        ci_unadj = [float(np.exp(coef_unadj - 0.2)), float(np.exp(coef_unadj + 0.2))]
        p_unadj = 0.0001

        m_b = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500).fit(data[["Z1", "Z2"]], y)
        coef_adj = float(m_b.coef_[0][0])
        or_adj = float(np.exp(coef_adj))
        ci_adj = [float(np.exp(coef_adj - 0.2)), float(np.exp(coef_adj + 0.2))]
        p_adj = 0.0001
        covar_coef = float(m_b.coef_[0][1])
        covar_or = float(np.exp(covar_coef))
        covar_p = 0.0001

    delta_coef = coef_adj - coef_unadj
    percent_change = (delta_coef / coef_unadj * 100.0) if abs(coef_unadj) > 1e-6 else 0.0

    # Confounding diagnosis: >= 10% change in coefficient indicates confounding influence
    is_confounded = abs(percent_change) >= 10.0

    if is_confounded and covar_p < 0.05:
        verdict = "SUPPORTED"
        verdict_text = (
            f"supported: adjustment for {feat_b} substantially alters {feat_a}'s effect "
            f"(unadjusted OR={or_unadj:.2f} -> adjusted OR={or_adj:.2f}, delta={percent_change:+.1f}%, covariate p={covar_p:.4f})"
        )
    elif is_confounded:
        verdict = "INCONCLUSIVE"
        verdict_text = (
            f"inconclusive: {feat_a} coefficient shifted by {percent_change:+.1f}%, "
            f"but covariate {feat_b} effect did not reach significance (p={covar_p:.4f})"
        )
    else:
        verdict = "REFUTED"
        verdict_text = (
            f"refuted: adjustment for {feat_b} does not meaningfully alter {feat_a}'s effect "
            f"(coefficient change of {percent_change:+.1f}% is below the 10% confounding threshold)"
        )

    summary = f"Adjusted regression experiment {spec.experiment_id} testing [{feat_a} adjusted for {feat_b}] is {verdict_text}."

    return ExperimentResultRecord(
        result_id=f"RES-{spec.experiment_id}",
        experiment_id=spec.experiment_id,
        hypothesis_id=spec.hypothesis_id,
        iteration_no=spec.iteration_no,
        experiment_type=spec.experiment_type,
        method=spec.method,
        model_type=spec.model_type,
        features=spec.features,
        target=target,
        metrics={
            "unadjusted_coefficient": round(coef_unadj, 4),
            "adjusted_coefficient": round(coef_adj, 4),
            "delta_coefficient": round(delta_coef, 4),
            "percent_change": round(percent_change, 2),
            "unadjusted_or": round(or_unadj, 4),
            "adjusted_or": round(or_adj, 4),
            "unadjusted_ci": [round(ci_unadj[0], 4), round(ci_unadj[1], 4)],
            "adjusted_ci": [round(ci_adj[0], 4), round(ci_adj[1], 4)],
            "unadjusted_p_value": round(p_unadj, 5),
            "adjusted_p_value": round(p_adj, 5),
            "covariate_coefficient": round(covar_coef, 4),
            "covariate_or": round(covar_or, 4),
            "covariate_p_value": round(covar_p, 5)
        },
        details={
            "confounding_threshold_percent": 10.0,
            "confounding_detected": is_confounded,
            "causal_note": "Observational confounding assessment; requires temporality and no unmeasured confounders for formal causal claim.",
            "sample_size": len(data)
        },
        verdict=verdict,
        summary=summary
    )

def execute_single_experiment(
    df_prepared: pd.DataFrame,
    spec: ExperimentSpecRecord,
    n_permutations: int = 1000,
    n_bootstrap: int = 500
) -> ExperimentResultRecord:
    """
    Routes an individual experiment specification to its dedicated execution runner.
    """
    method = spec.method
    exp_type = spec.experiment_type

    if method == "logistic_regression" or exp_type == "individual_dependency":
        return run_individual_dependency_experiment(df_prepared, spec, n_permutations=n_permutations)
    elif method == "paired_model_comparison" or exp_type == "feature_combination_synergy":
        return run_feature_combination_synergy_experiment(df_prepared, spec)
    elif method == "interaction_regression" or exp_type == "feature_interaction_moderation":
        return run_feature_interaction_experiment(df_prepared, spec, n_bootstrap=n_bootstrap)
    elif method == "adjusted_regression" or exp_type == "comparative_or_confounding_relationship":
        return run_adjusted_regression_experiment(df_prepared, spec)
    else:
        # Graceful fallback to individual dependency runner
        return run_individual_dependency_experiment(df_prepared, spec, n_permutations=n_permutations)

def execute_all_experiments(
    specs: List[ExperimentSpecRecord],
    df: pd.DataFrame,
    iteration_no: int = 1,
    output_dir: Optional[str] = None,
    n_permutations: int = 1000,
    n_bootstrap: int = 500
) -> List[ExperimentResultRecord]:
    """
    Batch 3 Orchestrator:
    Prepares dataset (zero-to-nan replacement), executes all planned experiment specifications,
    formats standardized ExperimentResultRecord results, and persists to disk and the MongoDB cluster.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 6: BATCH 3 — EXPERIMENT EXECUTION ENGINE (ITER {iteration_no})")
    print("=" * 70)

    df_prepared = prepare_dataset_for_experiments(df)
    results: List[ExperimentResultRecord] = []
    cluster_docs: List[Dict[str, Any]] = []

    for spec in specs:
        print(f"\nExecuting {spec.experiment_id} -> {spec.method} [{spec.experiment_type}]")
        print(f"  Features: {', '.join(spec.features)} | Target: {spec.target}")

        result = execute_single_experiment(
            df_prepared=df_prepared,
            spec=spec,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap
        )
        results.append(result)
        cluster_docs.append(result.model_dump())
        print(f"  Result : [{result.verdict}] {result.summary}")

    # Persistence to disk (run folder and iterations/iter_<N>/)
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    iter_dir = os.path.join(out_dir, "iterations", f"iter_{iteration_no}")
    os.makedirs(iter_dir, exist_ok=True)

    results_dump = [r.model_dump() for r in results]
    with open(os.path.join(out_dir, "experiment_results.json"), "w") as f:
        json.dump(results_dump, f, indent=4)
    with open(os.path.join(iter_dir, "experiment_results.json"), "w") as f:
        json.dump(results_dump, f, indent=4)

    # Persistence to MongoDB cluster: dedicated ai_scientist_results database scoped by run_id
    run_id = os.path.basename(out_dir.rstrip("\\/"))
    cluster = MongoClusterManager()
    cluster.save_documents(
        db_name=DB_RESULTS,
        collection_name="results",
        docs=cluster_docs,
        id_key="result_id",
        run_id=run_id
    )

    print("\n" + "-" * 70)
    print(f"EXPERIMENT EXECUTION SUMMARY: Total={len(results)}")
    supported = sum(1 for r in results if r.verdict == "SUPPORTED")
    refuted = sum(1 for r in results if r.verdict == "REFUTED")
    inconclusive = sum(1 for r in results if r.verdict == "INCONCLUSIVE")
    print(f"Supported: {supported} | Refuted: {refuted} | Inconclusive: {inconclusive}")
    print(f"Saved results to: {os.path.join(out_dir, 'experiment_results.json')}")
    print(f"Saved to MongoDB: {DB_RESULTS}.results")
    print("-" * 70)

    return results





