import pytest
from search.prm import StepVerifier, StepBeamSearch


def test_step_verifier_valid_arithmetic():
    verifier = StepVerifier()
    step = "First, calculate total items: 5 * 12 = 60 items."
    valid, score, msg = verifier.verify_arithmetic_step(step)
    assert valid is True
    assert score > 0.9


def test_step_verifier_flawed_arithmetic():
    verifier = StepVerifier()
    step = "Then, subtract items: 60 - 15 = 50 items remaining."
    valid, score, msg = verifier.verify_arithmetic_step(step)
    assert valid is False
    assert score < 0.2
    assert "Arithmetic error detected" in msg


def test_step_verifier_trajectory_scoring():
    verifier = StepVerifier()
    steps = [
        "Let x be the initial number.",
        "We know 2 * x = 20.",
        "Then x = 10.",
        "Add 5: 10 + 5 = 15."
    ]
    report = verifier.score_trajectory(steps)
    assert report["is_valid"] is True
    assert report["min_score"] > 0.8
    assert report["first_error_index"] is None


def test_step_verifier_trajectory_detects_error():
    verifier = StepVerifier()
    steps = [
        "We start with 100 dollars.",
        "We spend 20 dollars: 100 - 20 = 70 dollars left.",  # Flawed step
        "Then we add 10 dollars."
    ]
    report = verifier.score_trajectory(steps)
    assert report["is_valid"] is False
    assert report["first_error_index"] == 1  # Exactly step index 1


def test_step_beam_search():
    verifier = StepVerifier()
    beam_search = StepBeamSearch(step_verifier=verifier, beam_width=2)
    
    # Mock generator proposing 1 correct and 1 flawed step at each depth
    def mock_generator(problem: str, steps: list):
        if len(steps) >= 2:
            return []  # Terminal
        if len(steps) == 0:
            return ["Step 1: 10 + 10 = 20", "Step 1: 10 + 10 = 25"]
        if len(steps) == 1:
            return ["Step 2: 20 * 2 = 40", "Step 2: 20 * 2 = 50"]
        return []

    res = beam_search.search("Compute final value", mock_generator, max_depth=3)
    assert len(res["best_trajectory"]) == 2
    assert res["best_trajectory"][0] == "Step 1: 10 + 10 = 20"
    assert res["best_trajectory"][1] == "Step 2: 20 * 2 = 40"
    assert res["score"] > 0.8
