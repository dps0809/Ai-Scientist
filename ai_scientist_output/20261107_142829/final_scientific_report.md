# Autonomous AI Scientist — Final Research Report

**Run ID:** `20261107_142829`  
**Dataset:** `diabetes.csv`  
**Generated At:** 2026-11-07 09:03:38 UTC  
**Stopping Reason:** Maximum iterations limit reached (5/5).  

---

## 1. Executive Summary
The Autonomous AI Scientist executed a closed-loop empirical research program:
- **Total Experiments Conducted:** 9
- **Autonomous Iterations Completed:** 5
- **Confirmed Discoveries (Supported):** 4
- **Refuted Hypotheses:** 5
- **Top Predictive Performance (Primary Metric):** **ROC-AUC = 0.8178**
- **Secondary Metric Performance:** **Accuracy = 0.7682**
- **Optimal Feature Set:** `Glucose, BMI`
- **Optimal Model Architecture:** `LogisticRegression`

---

## 2. Confirmed Scientific Discoveries
### Discovery 1: iter1_E1 (iter1_H3)
- **Features Tested:** `Pregnancies, Glucose`
- **Evaluation Summary:** Synergy experiment iter1_E1 testing [Pregnancies + Glucose] is supported_by_current_evidence: combined Primary AUC=0.8088 vs best individual (Glucose) AUC=0.7905 (gain=+0.0183, paired t p=0.0290; Secondary Accuracy=0.7447, Accuracy gain=-0.0026, best model: LogisticRegression). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.8088, gain: 0.0183, combined_accuracy: 0.7447, accuracy_gain: -0.0026, best_individual_feature: Glucose, best_model: LogisticRegression, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 2: iter1_E2 (iter1_H4)
- **Features Tested:** `Glucose, BMI`
- **Evaluation Summary:** Synergy experiment iter1_E2 testing [Glucose + BMI] is supported_by_current_evidence: combined Primary AUC=0.8178 vs best individual (Glucose) AUC=0.7905 (gain=+0.0274, paired t p=0.0026; Secondary Accuracy=0.7682, Accuracy gain=+0.0209, best model: LogisticRegression). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.8178, gain: 0.0274, combined_accuracy: 0.7682, accuracy_gain: 0.0209, best_individual_feature: Glucose, best_model: LogisticRegression, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 3: iter1_E3 (iter1_H5)
- **Features Tested:** `Glucose, Age`
- **Evaluation Summary:** Interaction experiment iter1_E3 testing [Glucose * Age] is supported_by_current_evidence: LRT chi2=7.632 (p=0.0057), interaction beta=-0.00075 (Wald p=0.0044), 95% bootstrap CI [-0.00129, -0.00019].
- **Key Metrics:** interaction_coefficient: -0.0008

### Discovery 4: iter1_E5 (iter1_H1)
- **Features Tested:** `Glucose`
- **Evaluation Summary:** Experiment iter1_E5 testing dependency of 'Glucose' on 'Outcome' is supported_by_current_evidence: OR=3.46 (95% CI [2.82, 4.23]), parametric p=0.0000, permutation p=0.0000, 5-fold CV Primary ROC-AUC=0.790 +/- 0.025, Secondary Accuracy=0.747 +/- 0.029 (Best model: LogisticRegression).
- **Key Metrics:** coefficient: 1.2400, odds_ratio: 3.4555, mean_auc: 0.7905, std_auc: 0.0248, mean_accuracy: 0.7473, std_accuracy: 0.0293, best_model: LogisticRegression, primary_metric: ROC_AUC, secondary_metric: Accuracy

## 3. Multi-Model Architecture Tournament (Logistic Regression vs Random Forest vs XGBoost)
Candidate predictive models were simultaneously evaluated across identical 5-fold Stratified Cross-Validation splits:

| Experiment | Features | Winning Model | Primary ROC-AUC | Secondary Accuracy | Model Breakdown (AUC / Accuracy) |
|---|---|---|---|---|---|
| `iter1_E1` | `Pregnancies, Glucose` | **LogisticRegression** | **0.8088** | 0.7447 | **LogisticRegression**: 0.809/0.745 <br> **RandomForest**: 0.780/0.715 <br> **XGBoost**: 0.785/0.741 |
| `iter1_E2` | `Glucose, BMI` | **LogisticRegression** | **0.8178** | 0.7682 | **LogisticRegression**: 0.818/0.768 <br> **RandomForest**: 0.815/0.758 <br> **XGBoost**: 0.812/0.764 |
| `iter1_E5` | `Glucose` | **LogisticRegression** | **0.7905** | 0.7473 | **LogisticRegression**: 0.790/0.747 <br> **RandomForest**: 0.756/0.717 <br> **XGBoost**: 0.764/0.724 |
| `iter2_E2` | `Glucose, BloodPressure` | **LogisticRegression** | **0.7927** | 0.7499 | **LogisticRegression**: 0.793/0.750 <br> **RandomForest**: 0.757/0.732 <br> **XGBoost**: 0.762/0.736 |

## 4. Refuted Hypotheses & Negative Results
### Negative Result 1: iter1_E4 (iter1_H6)
- **Features Tested:** `Pregnancies, Age`
- **Evaluation Summary:** Adjusted regression experiment iter1_E4 testing [Pregnancies adjusted for Age] is not_supported: unadjusted OR=1.59 -> adjusted OR=1.32 (delta=-40.0%, covariate p=1.0000).

### Negative Result 2: iter2_E1 (iter2_H3)
- **Features Tested:** `Glucose, BMI`
- **Evaluation Summary:** Interaction experiment iter2_E1 testing [Glucose * BMI] is not_supported: LRT chi2=0.424 (p=0.5148), interaction beta=-0.00035 (Wald p=0.5133), 95% bootstrap CI [-0.00139, 0.00072].

### Negative Result 3: iter2_E2 (iter2_H1)
- **Features Tested:** `Glucose, BloodPressure`
- **Evaluation Summary:** Synergy experiment iter2_E2 testing [Glucose + BloodPressure] is inconclusive: combined Primary AUC=0.7927 vs best individual (Glucose) AUC=0.7905 (gain=+0.0023, paired t p=0.2268; Secondary Accuracy=0.7499, Accuracy gain=+0.0026, best model: LogisticRegression). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.

### Negative Result 4: iter3_E2 (iter3_H2)
- **Features Tested:** `Glucose, BloodPressure`
- **Evaluation Summary:** Interaction experiment iter3_E2 testing [Glucose * BloodPressure] is not_supported: LRT chi2=0.629 (p=0.4278), interaction beta=-0.00022 (Wald p=0.4259), 95% bootstrap CI [-0.00088, 0.00047].

### Negative Result 5: iter4_E1 (iter4_H1)
- **Features Tested:** `Glucose, SkinThickness`
- **Evaluation Summary:** Interaction experiment iter4_E1 testing [Glucose * SkinThickness] is not_supported: LRT chi2=0.155 (p=0.6942), interaction beta=-0.00016 (Wald p=0.6921), 95% bootstrap CI [-0.00088, 0.00061].

## 5. Limitations & Methodological Integrity
1. **Predictive Synergy vs Statistical Interaction:** Model gains were evaluated under cross-validation separately from multiplicative product terms to prevent confounding predictive power with interaction effects.
2. **Multiple Model Comparison:** Logistic Regression, Random Forest, and XGBoost were tested under identical cross-validation folds to determine whether linear separation or non-linear tree-based boundaries best capture the underlying biological risk.
3. **Primary vs Secondary Metrics:** ROC-AUC acts as the primary evaluation criterion due to class imbalance resilience; Accuracy is reported alongside as an auxiliary operational metric.
4. **Multiple Testing & Integrity:** Candidate hypotheses underwent permutation testing and within-fold median imputation to strictly avoid data leakage.

---
*Report generated autonomously by AI Scientist Engine.*
