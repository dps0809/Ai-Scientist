import pytest
from modules.hypothesis_registry import HypothesisRegistry
from modules.schemas import HypothesisRecord

def test_registry_prevents_duplicate_hypotheses(tmp_path):
    registry_file = tmp_path / "hypo_reg.json"
    registry = HypothesisRegistry(registry_path=str(registry_file))
    
    h1 = HypothesisRecord(
        hypothesis_id="H1",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose concentration has strong positive association with outcome.",
        features_involved=["Glucose"],
        evidence_from_data="r=0.4666",
        proposed_experiment="Logistic regression"
    )
    is_new = registry.register_hypothesis(h1)
    assert is_new is True
    assert len(registry.get_all_hypotheses()) == 1
    
    # Attempt duplicate in iteration 2 with same feature and hypothesis type
    h1_duplicate = HypothesisRecord(
        hypothesis_id="H1",
        iteration_no=2,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose is positively associated with diabetes outcome.",
        features_involved=["Glucose"],
        evidence_from_data="r=0.4666",
        proposed_experiment="Logistic regression"
    )
    assert registry.is_duplicate(h1_duplicate) is True
    is_new_again = registry.register_hypothesis(h1_duplicate)
    assert is_new_again is False

def test_registry_allows_different_features(tmp_path):
    registry_file = tmp_path / "hypo_reg.json"
    registry = HypothesisRegistry(registry_path=str(registry_file))
    
    h1 = HypothesisRecord(
        hypothesis_id="H1",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose concentration has strong positive association.",
        features_involved=["Glucose"],
        evidence_from_data="r=0.4666",
        proposed_experiment="Logistic regression"
    )
    h2 = HypothesisRecord(
        hypothesis_id="H2",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="BMI shows moderate positive association.",
        features_involved=["BMI"],
        evidence_from_data="r=0.2927",
        proposed_experiment="Logistic regression"
    )
    assert registry.register_hypothesis(h1) is True
    assert registry.register_hypothesis(h2) is True
    assert len(registry.get_all_hypotheses()) == 2
    
    # Summary for prompt
    summary = registry.get_past_hypotheses_prompt_summary()
    assert "Glucose" in summary
    assert "BMI" in summary
