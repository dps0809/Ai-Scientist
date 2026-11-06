import pytest
from modules.schemas import HypothesisRecord, RefinedHypothesisRecord
from modules.experiment_selector import rank_and_select_candidates, score_candidate_promise


def test_rank_and_select_candidates_prioritizes_valid_synergies():
    pool = [
        RefinedHypothesisRecord(
            hypothesis_id="iter2_H1",
            parent_hypothesis_id="iter1_H1",
            iteration_no=2,
            hypothesis="Glucose and UnknownChemical interact.",
            features_involved=["Glucose", "UnknownChemical"],
            proposed_experiment={"features": ["Glucose", "UnknownChemical"], "model": "statsmodels.Logit"}
        ),
        RefinedHypothesisRecord(
            hypothesis_id="iter2_H2",
            parent_hypothesis_id="iter1_H3",
            iteration_no=2,
            hypothesis="Glucose, BMI, and Age yield 3-feature predictive synergy.",
            features_involved=["Glucose", "BMI", "Age"],
            proposed_experiment={"features": ["Glucose", "BMI", "Age"], "model": "sklearn.linear_model.LogisticRegression"}
        ),
        RefinedHypothesisRecord(
            hypothesis_id="iter2_H3",
            parent_hypothesis_id="iter1_H2",
            iteration_no=2,
            hypothesis="Pregnancies alone predicts Outcome.",
            features_involved=["Pregnancies"],
            proposed_experiment={"features": ["Pregnancies"], "model": "statsmodels.Logit"}
        )
    ]

    valid_cols = ["Glucose", "BMI", "Age", "Pregnancies", "Outcome"]
    ranked = rank_and_select_candidates(
        candidate_pool=pool,
        valid_columns=valid_cols,
        memory_stats={"best_features": ["Glucose", "BMI"]},
        top_k=2
    )

    # iter2_H1 must be filtered out or heavily penalized because 'UnknownChemical' is not in dataset
    assert len(ranked) == 2
    assert ranked[0].hypothesis_id == "iter2_H2"
    assert ranked[1].hypothesis_id == "iter2_H3"


def test_score_candidate_promise_rejects_hallucinations():
    valid_cols = ["Glucose", "BMI", "Outcome"]
    hypo_hallucinated = HypothesisRecord(
        hypothesis_id="H_bad",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="HOMA-IR predicts outcome.",
        features_involved=["HOMA-IR"]
    )
    score = score_candidate_promise(hypo_hallucinated, valid_columns=valid_cols)
    assert score == 0.0

    hypo_good = HypothesisRecord(
        hypothesis_id="H_good",
        iteration_no=1,
        hypothesis_type="feature_combination_synergy",
        hypothesis="Glucose and BMI predict outcome.",
        features_involved=["Glucose", "BMI"]
    )
    score_good = score_candidate_promise(hypo_good, valid_columns=valid_cols, memory_stats={"best_features": ["Glucose"]})
    assert score_good > 1.0
