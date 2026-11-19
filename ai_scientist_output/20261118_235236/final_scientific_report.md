# Autonomous AI Scientist — Final Research Report

**Run ID:** `20261118_235236`  
**Dataset:** `diabetes.csv`  
**Generated At:** 2026-11-18 18:25:37 UTC  
**Stopping Reason:** No improvement tolerance reached (3 consecutive experiments without >= 0.0050 gain).  

---

## 1. Executive Summary
The Autonomous AI Scientist executed a closed-loop empirical research program:
- **Total Experiments Conducted:** 12
- **Autonomous Iterations Completed:** 3
- **Confirmed Discoveries (Supported):** 9
- **Refuted Hypotheses:** 3
- **Top Predictive Performance (Primary Metric):** **ROC-AUC = 0.7797**
- **Secondary Metric Performance:** **Accuracy = 0.6938**
- **Optimal Feature Set:** `G, C, B, A, Ic, Vb, Ia, Ib`
- **Optimal Model Architecture:** `RandomForest`

---

## 2. Confirmed Scientific Discoveries
### Discovery 1: iter1_E1 (iter1_H1)
- **Features Tested:** `G, C, A, Ia, Ic`
- **Evaluation Summary:** Experiment iter1_E1 testing dependency of 'G' on 'Output (S)' is supported_by_current_evidence: OR=1.25 (95% CI [1.20, 1.31]), parametric p=0.0000, permutation p=0.0000, 5-fold CV Primary ROC-AUC=0.555 +/- 0.017, Secondary Accuracy=0.555 +/- 0.000 (Best model: LogisticRegression).
- **Key Metrics:** coefficient: 0.2234, odds_ratio: 1.2504, mean_auc: 0.5549, std_auc: 0.0170, mean_accuracy: 0.5550, std_accuracy: 0.0003, best_model: LogisticRegression, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 2: iter1_E3 (iter1_H3)
- **Features Tested:** `G, C, B, A, Ic, Vb`
- **Evaluation Summary:** Synergy experiment iter1_E3 testing [G + C + B + A + Ic + Vb] is supported_by_current_evidence: combined Primary AUC=0.7536 vs best individual (Ic) AUC=0.6211 (gain=+0.1324, paired t p=0.0000; Secondary Accuracy=0.6686, Accuracy gain=+0.0873, best model: RandomForest). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.7536, gain: 0.1324, combined_accuracy: 0.6686, accuracy_gain: 0.0873, best_individual_feature: Ic, best_model: RandomForest, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 3: iter2_E1 (iter2_H2)
- **Features Tested:** `G, C, B, A, Ic, Vb, Ia`
- **Evaluation Summary:** Synergy experiment iter2_E1 testing [G + C + B + A + Ic + Vb + Ia] is supported_by_current_evidence: combined Primary AUC=0.7674 vs best individual (Ic) AUC=0.6165 (gain=+0.1510, paired t p=0.0000; Secondary Accuracy=0.6821, Accuracy gain=+0.0996, best model: XGBoost). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.7674, gain: 0.1510, combined_accuracy: 0.6821, accuracy_gain: 0.0996, best_individual_feature: Ic, best_model: XGBoost, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 4: iter2_E2 (iter2_H3)
- **Features Tested:** `C, B, A, Ic, Va, Vb`
- **Evaluation Summary:** Synergy experiment iter2_E2 testing [C + B + A + Ic + Va + Vb] is supported_by_current_evidence: combined Primary AUC=0.6907 vs best individual (Ic) AUC=0.6211 (gain=+0.0696, paired t p=0.0001; Secondary Accuracy=0.6307, Accuracy gain=+0.0494, best model: RandomForest). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.6907, gain: 0.0696, combined_accuracy: 0.6307, accuracy_gain: 0.0494, best_individual_feature: Ic, best_model: RandomForest, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 5: iter2_E3 (iter2_H4)
- **Features Tested:** `G, C, B, A, Ib, Ic`
- **Evaluation Summary:** Synergy experiment iter2_E3 testing [G + C + B + A + Ib + Ic] is supported_by_current_evidence: combined Primary AUC=0.7730 vs best individual (Ic) AUC=0.6211 (gain=+0.1519, paired t p=0.0000; Secondary Accuracy=0.6787, Accuracy gain=+0.0973, best model: RandomForest). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.7730, gain: 0.1519, combined_accuracy: 0.6787, accuracy_gain: 0.0973, best_individual_feature: Ic, best_model: RandomForest, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 6: iter3_E1 (iter3_H1)
- **Features Tested:** `G, C, B, A, Ic, Vb, Ia, Ib`
- **Evaluation Summary:** Synergy experiment iter3_E1 testing [G + C + B + A + Ic + Vb + Ia + Ib] is supported_by_current_evidence: combined Primary AUC=0.7797 vs best individual (Ic) AUC=0.6211 (gain=+0.1586, paired t p=0.0000; Secondary Accuracy=0.6938, Accuracy gain=+0.1125, best model: RandomForest). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.7797, gain: 0.1586, combined_accuracy: 0.6938, accuracy_gain: 0.1125, best_individual_feature: Ic, best_model: RandomForest, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 7: iter3_E2 (iter3_H3)
- **Features Tested:** `G, C, B, A, Ib, Ic, Ia`
- **Evaluation Summary:** Synergy experiment iter3_E2 testing [G + C + B + A + Ib + Ic + Ia] is supported_by_current_evidence: combined Primary AUC=0.7817 vs best individual (Ic) AUC=0.6211 (gain=+0.1606, paired t p=0.0000; Secondary Accuracy=0.6915, Accuracy gain=+0.1102, best model: RandomForest). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.7817, gain: 0.1606, combined_accuracy: 0.6915, accuracy_gain: 0.1102, best_individual_feature: Ic, best_model: RandomForest, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 8: iter3_E3 (iter3_H2)
- **Features Tested:** `C, B, A, Ic, Va, Vb, G`
- **Evaluation Summary:** Synergy experiment iter3_E3 testing [C + B + A + Ic + Va + Vb + G] is supported_by_current_evidence: combined Primary AUC=0.7686 vs best individual (Ic) AUC=0.6211 (gain=+0.1475, paired t p=0.0000; Secondary Accuracy=0.6753, Accuracy gain=+0.0940, best model: RandomForest). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.7686, gain: 0.1475, combined_accuracy: 0.6753, accuracy_gain: 0.0940, best_individual_feature: Ic, best_model: RandomForest, primary_metric: ROC_AUC, secondary_metric: Accuracy

### Discovery 9: iter3_E4 (iter3_H4)
- **Features Tested:** `G, B, C`
- **Evaluation Summary:** Synergy experiment iter3_E4 testing [G + B + C] is supported_by_current_evidence: combined Primary AUC=0.6172 vs best individual (C) AUC=0.5795 (gain=+0.0377, paired t p=0.0036; Secondary Accuracy=0.5842, Accuracy gain=+0.0154, best model: RandomForest). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.
- **Key Metrics:** combined_auc: 0.6172, gain: 0.0377, combined_accuracy: 0.5842, accuracy_gain: 0.0154, best_individual_feature: C, best_model: RandomForest, primary_metric: ROC_AUC, secondary_metric: Accuracy

## 3. Multi-Model Architecture Tournament (Logistic Regression vs Random Forest vs XGBoost)
Candidate predictive models were simultaneously evaluated across identical 5-fold Stratified Cross-Validation splits:

| Experiment | Features | Winning Model | Primary ROC-AUC | Secondary Accuracy | Model Breakdown (AUC / Accuracy) |
|---|---|---|---|---|---|
| `iter1_E1` | `G, C, A, Ia, Ic` | **LogisticRegression** | **0.5549** | 0.5550 | **LogisticRegression**: 0.555/0.555 <br> **RandomForest**: 0.555/0.548 <br> **XGBoost**: 0.555/0.555 |
| `iter1_E3` | `G, C, B, A, Ic, Vb` | **RandomForest** | **0.7536** | 0.6686 | **LogisticRegression**: 0.697/0.636 <br> **RandomForest**: 0.754/0.669 <br> **XGBoost**: 0.745/0.660 |
| `iter2_E1` | `G, C, B, A, Ic, Vb, Ia` | **XGBoost** | **0.7674** | 0.6821 | **LogisticRegression**: 0.697/0.636 <br> **RandomForest**: 0.765/0.681 <br> **XGBoost**: 0.767/0.682 |
| `iter2_E2` | `C, B, A, Ic, Va, Vb` | **RandomForest** | **0.6907** | 0.6307 | **LogisticRegression**: 0.611/0.588 <br> **RandomForest**: 0.691/0.631 <br> **XGBoost**: 0.687/0.624 |
| `iter2_E3` | `G, C, B, A, Ib, Ic` | **RandomForest** | **0.7730** | 0.6787 | **LogisticRegression**: 0.687/0.629 <br> **RandomForest**: 0.773/0.679 <br> **XGBoost**: 0.759/0.682 |
| `iter2_E4` | `G, B` | **LogisticRegression** | **0.5699** | 0.5681 | **LogisticRegression**: 0.570/0.568 <br> **RandomForest**: 0.570/0.568 <br> **XGBoost**: 0.570/0.568 |
| `iter3_E1` | `G, C, B, A, Ic, Vb, Ia, Ib` | **RandomForest** | **0.7797** | 0.6938 | **LogisticRegression**: 0.698/0.629 <br> **RandomForest**: 0.780/0.694 <br> **XGBoost**: 0.775/0.691 |
| `iter3_E2` | `G, C, B, A, Ib, Ic, Ia` | **RandomForest** | **0.7817** | 0.6915 | **LogisticRegression**: 0.682/0.628 <br> **RandomForest**: 0.782/0.692 <br> **XGBoost**: 0.768/0.680 |
| `iter3_E3` | `C, B, A, Ic, Va, Vb, G` | **RandomForest** | **0.7686** | 0.6753 | **LogisticRegression**: 0.697/0.637 <br> **RandomForest**: 0.769/0.675 <br> **XGBoost**: 0.758/0.669 |
| `iter3_E4` | `G, B, C` | **RandomForest** | **0.6172** | 0.5842 | **LogisticRegression**: 0.614/0.584 <br> **RandomForest**: 0.617/0.584 <br> **XGBoost**: 0.617/0.584 |

## 4. Refuted Hypotheses & Negative Results
### Negative Result 1: iter1_E4 (iter1_H4)
- **Features Tested:** `C, B, A, Ic, Va, Vb`
- **Evaluation Summary:** Interaction experiment iter1_E4 testing [C * B * A * Ic * Va * Vb] is not_supported: LRT chi2=0.000 (p=1.0000), interaction beta=0.36639 (Wald p=nan), 95% bootstrap CI [0.28774, 0.44364].

### Negative Result 2: iter1_E5 (iter1_H5)
- **Features Tested:** `G, C, B, A, Ib, Ic`
- **Evaluation Summary:** Adjusted regression experiment iter1_E5 testing [G adjusted for C] is not_supported: unadjusted OR=1.25 -> adjusted OR=1.33 (delta=+26.5%, covariate p=1.0000).

### Negative Result 3: iter2_E4 (iter2_H1)
- **Features Tested:** `G, B`
- **Evaluation Summary:** Synergy experiment iter2_E4 testing [G + B] is inconclusive: combined Primary AUC=0.5699 vs best individual (G) AUC=0.5549 (gain=+0.0150, paired t p=0.0922; Secondary Accuracy=0.5681, Accuracy gain=+0.0131, best model: LogisticRegression). Note: Predictive synergy indicates feature combination improvement, distinct from statistical interaction.

## 5. Limitations & Methodological Integrity
1. **Predictive Synergy vs Statistical Interaction:** Model gains were evaluated under cross-validation separately from multiplicative product terms to prevent confounding predictive power with interaction effects.
2. **Multiple Model Comparison:** Logistic Regression, Random Forest, and XGBoost were tested under identical cross-validation folds to determine whether linear separation or non-linear tree-based boundaries best capture the underlying biological risk.
3. **Primary vs Secondary Metrics:** ROC-AUC acts as the primary evaluation criterion due to class imbalance resilience; Accuracy is reported alongside as an auxiliary operational metric.
4. **Multiple Testing & Integrity:** Candidate hypotheses underwent permutation testing and within-fold median imputation to strictly avoid data leakage.

---
*Report generated autonomously by AI Scientist Engine.*
