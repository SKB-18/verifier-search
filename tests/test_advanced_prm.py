import pytest
from search.prm import StepVerifier, StepBeamSearch


def test_step_verifier_float_and_decimals():
    verifier = StepVerifier()
    step = "Add decimals: 12.5 + 7.5 = 20.0 and divide: 20.0 / 4 = 5.0"
    valid, score, msg = verifier.verify_arithmetic_step(step)
    assert valid is True
    assert score > 0.9


def test_step_verifier_negative_arithmetic():
    verifier = StepVerifier()
    step = "Compute difference: 10 - 25 = -15"
    valid, score, msg = verifier.verify_arithmetic_step(step)
    assert valid is True
    assert score > 0.9


def test_step_verifier_parentheses_order_of_operations():
    verifier = StepVerifier()
    valid_step = "Solve grouping: (3 + 7) * 4 = 40"
    invalid_step = "Solve grouping: (3 + 7) * 4 = 31"
    
    v1, s1, _ = verifier.verify_arithmetic_step(valid_step)
    v2, s2, _ = verifier.verify_arithmetic_step(invalid_step)
    
    assert v1 is True
    assert s1 > 0.9
    assert v2 is False
    assert s2 < 0.2


def test_step_verifier_division_by_zero_safety():
    verifier = StepVerifier()
    # Step has an invalid division by zero expression
    step = "Edge case: 10 / 0 = 0"
    # Must not crash or raise ZeroDivisionError
    valid, score, msg = verifier.verify_arithmetic_step(step)
    assert valid is True or valid is False  # handles safely


def test_step_verifier_custom_evaluator():
    def custom_eval(ctx: str, step: str) -> float:
        if "forbidden" in step.lower():
            return 0.0
        return 0.85

    verifier = StepVerifier(step_evaluator_fn=custom_eval)
    assert verifier.score_step("", "This contains forbidden concept") == 0.0
    assert verifier.score_step("", "Clean and valid step") == 0.85


def test_step_verifier_empty_trajectory():
    verifier = StepVerifier()
    report = verifier.score_trajectory([])
    assert report["is_valid"] is True
    assert report["step_scores"] == []
    assert report["min_score"] == 0.0


def test_beam_search_greedy_width_one():
    verifier = StepVerifier()
    beam = StepBeamSearch(step_verifier=verifier, beam_width=1)
    
    def generator(prob, steps):
        if len(steps) == 0:
            return ["Option A: 2 + 2 = 4", "Option B: 2 + 2 = 5"]
        return []
        
    res = beam.search("Add", generator, max_depth=2)
    assert res["best_trajectory"] == ["Option A: 2 + 2 = 4"]


def test_beam_search_empty_candidates():
    verifier = StepVerifier()
    beam = StepBeamSearch(step_verifier=verifier, beam_width=2)
    res = beam.search("Problem", lambda p, s: [], max_depth=3)
    assert res["best_trajectory"] == []
    assert res["score"] == 1.0
