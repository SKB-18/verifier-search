import pytest
from search.best_of_n import run_best_of_n
from search.mcts import run_mcts, MCTSNode
from search.scaling import TestTimeComputeBenchmarker


def test_best_of_n_empty():
    res = run_best_of_n("prompt", 5, lambda p, n: [], lambda p, c: 1.0)
    assert res is None


def test_best_of_n_negative_scores():
    candidates = ["candidate_bad", "candidate_worse", "candidate_least_bad"]
    scores = {"candidate_bad": -5.0, "candidate_worse": -10.0, "candidate_least_bad": -1.0}
    
    res = run_best_of_n(
        prompt="prompt",
        n=3,
        generator_fn=lambda p, n: candidates,
        verifier_fn=lambda p, c: scores[c]
    )
    assert res == "candidate_least_bad"


def test_mcts_root_node_uct():
    root = MCTSNode(state="root")
    # visits == 0 returns inf
    assert root.uct_score() == float('inf')


def test_mcts_deterministic_tree():
    # Tree where state is integer: child is state + 1 or state + 2
    # Reward is 1.0 if state == 5, else 0.0
    def gen_fn(state):
        if state >= 5:
            return []
        return [state + 1, state + 2]

    def roll_fn(state):
        return 1.0 if state == 5 else 0.1

    best_next = run_mcts(
        initial_state=3,
        num_simulations=50,
        generate_fn=gen_fn,
        rollout_fn=roll_fn
    )
    # Moving from 3 to 5 directly (+2) yields highest instant reward
    assert best_next in [4, 5]


def test_scaling_benchmarker_all_correct():
    benchmarker = TestTimeComputeBenchmarker(
        generator_fn=lambda p, n: [f"sol_{i}" for i in range(n)],
        verifier_fn=lambda p, c: 1.0,
        ground_truth_fn=lambda p, c: True
    )
    res = benchmarker.benchmark_budget_scaling(
        problems=["prob1", "prob2"],
        budgets=[1, 2, 4]
    )
    for b in [1, 2, 4]:
        assert res["scaling_curve"][b]["accuracy"] == 1.0


def test_scaling_benchmarker_monotonic_improvement():
    # Model that produces correct answer 25% of the time
    import random
    
    def generator(prob, n):
        # Deterministic simulation: first 3 wrong, 4th right
        cands = [f"wrong_{i}" for i in range(max(1, n - 1))]
        if n >= 4:
            cands.append("correct_answer")
        return cands

    def verifier(prob, cand):
        return 0.9 if cand == "correct_answer" else 0.2

    def ground_truth(prob, cand):
        return cand == "correct_answer"

    benchmarker = TestTimeComputeBenchmarker(
        generator_fn=generator,
        verifier_fn=verifier,
        ground_truth_fn=ground_truth
    )
    
    res = benchmarker.benchmark_budget_scaling(
        problems=["math_problem"],
        budgets=[1, 2, 4]
    )
    # N=1 accuracy is 0, N=4 accuracy is 1.0
    assert res["scaling_curve"][1]["accuracy"] == 0.0
    assert res["scaling_curve"][4]["accuracy"] == 1.0
