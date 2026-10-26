import pytest
from search.scaling import TestTimeComputeBenchmarker


def test_test_time_compute_scaling():
    # Generator produces candidates where 1 in 4 candidates is correct
    def generator_fn(prompt: str, n: int):
        candidates = []
        for i in range(n):
            if i % 3 == 0:
                candidates.append("Step 1: 5 * 5 = 25. Final Answer: 25")
            else:
                candidates.append("Step 1: 5 * 5 = 20. Final Answer: 20")
        return candidates

    # Verifier detects the correct arithmetic
    def verifier_fn(prompt: str, candidate: str):
        if "25" in candidate and "5 * 5 = 25" in candidate:
            return 1.0
        return 0.0

    def ground_truth_fn(prompt: str, candidate: str):
        return "Final Answer: 25" in candidate

    benchmarker = TestTimeComputeBenchmarker(
        generator_fn=generator_fn,
        verifier_fn=verifier_fn,
        ground_truth_fn=ground_truth_fn
    )

    problems = ["Problem 1", "Problem 2", "Problem 3"]
    results = benchmarker.benchmark_budget_scaling(problems, budgets=[1, 2, 4])
    
    curve = results["scaling_curve"]
    assert 1 in curve
    assert 4 in curve
    # With N=4, probability of finding the correct answer is 100%
    assert curve[4]["accuracy"] == 1.0
