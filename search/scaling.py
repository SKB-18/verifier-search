from typing import List, Dict, Any, Callable
from search.best_of_n import run_best_of_n


class TestTimeComputeBenchmarker:
    __test__ = False
    """
    Evaluates test-time compute scaling laws:
    Demonstrates how performance scales as candidate budget N increases
    from N=1 to N=16 using verifier-guided rejection sampling.
    """

    def __init__(
        self,
        generator_fn: Callable[[str, int], List[str]],
        verifier_fn: Callable[[str, str], float],
        ground_truth_fn: Callable[[str, str], bool]
    ):
        self.generator_fn = generator_fn
        self.verifier_fn = verifier_fn
        self.ground_truth_fn = ground_truth_fn

    def benchmark_budget_scaling(
        self, 
        problems: List[str], 
        budgets: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Any]:
        """
        Measures accuracy curves across different values of N.
        """
        results_by_budget: Dict[int, Dict[str, Any]] = {}

        for n in budgets:
            correct_count = 0
            total_count = len(problems)

            for prob in problems:
                best_sol = run_best_of_n(
                    prompt=prob,
                    n=n,
                    generator_fn=self.generator_fn,
                    verifier_fn=self.verifier_fn
                )
                if best_sol and self.ground_truth_fn(prob, best_sol):
                    correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0.0
            results_by_budget[n] = {
                "n_budget": n,
                "correct": correct_count,
                "total": total_count,
                "accuracy": round(accuracy, 4)
            }

        return {
            "num_problems": len(problems),
            "scaling_curve": results_by_budget
        }
