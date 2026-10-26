from typing import List, Dict, Any, Tuple, Optional, Callable
import re
import math


class StepVerifier:
    """
    Process Reward Model (PRM) Step Verifier.
    Evaluates individual reasoning steps in a Chain-of-Thought (CoT) trajectory.
    Detects early arithmetic errors, false assumptions, and logical inconsistencies.
    """
    def __init__(self, step_evaluator_fn: Optional[Callable[[str, str], float]] = None):
        self.step_evaluator_fn = step_evaluator_fn

    def verify_arithmetic_step(self, step_text: str) -> Tuple[bool, float, str]:
        """
        Parses equations in the format 'X = Y' or 'A + B = C' within a step
        and validates mathematical correctness.
        """
        # Look for equations like: 15 * 4 = 60, 10 - 25 = -15
        equations = re.findall(r"([0-9\.\+\-\*\/\s\(\)]+)\s*=\s*([\-]?[0-9\.]+)", step_text)
        if not equations:
            return True, 0.9, "No explicit arithmetic equations found; plausible logic."

        for expr, expected_str in equations:
            expr_clean = expr.strip()
            # Safety check: only allow basic arithmetic tokens
            if not re.match(r"^[0-9\.\+\-\*\/\s\(\)]+$", expr_clean):
                continue
            try:
                actual_val = eval(expr_clean, {"__builtins__": None}, {})
                expected_val = float(expected_str.strip())
                if abs(actual_val - expected_val) > 1e-4:
                    return False, 0.1, f"Arithmetic error detected: '{expr_clean}' evaluates to {actual_val}, but step states {expected_val}."
            except Exception:
                continue

        return True, 0.95, "Arithmetic verified."

    def score_step(self, previous_context: str, current_step: str) -> float:
        """Scores a single reasoning step in context [0.0 to 1.0]."""
        if self.step_evaluator_fn:
            return self.step_evaluator_fn(previous_context, current_step)

        # Built-in heuristic rule-based verifier
        valid, score, _ = self.verify_arithmetic_step(current_step)
        if not valid:
            return score
        
        # Check for empty / vacuous step
        if len(current_step.strip()) < 5:
            return 0.2

        return score

    def score_trajectory(self, steps: List[str]) -> Dict[str, Any]:
        """
        Evaluates a complete sequence of reasoning steps.
        Computes step-level scores and aggregate trajectory score (minimum / product).
        """
        context = ""
        step_scores = []
        is_valid = True
        error_step_index = None

        for idx, step in enumerate(steps):
            score = self.score_step(context, step)
            step_scores.append(score)
            if score < 0.5 and is_valid:
                is_valid = False
                error_step_index = idx
            context += f"\nStep {idx+1}: {step}"

        # Aggregate PRM score: minimum step score (bottleneck policy)
        min_score = min(step_scores) if step_scores else 0.0
        # Average step score
        avg_score = sum(step_scores) / len(step_scores) if step_scores else 0.0

        return {
            "step_scores": step_scores,
            "min_score": round(min_score, 3),
            "avg_score": round(avg_score, 3),
            "is_valid": is_valid,
            "first_error_index": error_step_index
        }


class StepBeamSearch:
    """
    Step-Level Beam Search guided by a Process Reward Model (PRM).
    Prunes invalid reasoning paths early and explores the top-K highest-confidence branches.
    """
    def __init__(self, step_verifier: StepVerifier, beam_width: int = 2):
        self.verifier = step_verifier
        self.beam_width = beam_width

    def search(
        self,
        problem: str,
        step_generator_fn: Callable[[str, List[str]], List[str]],
        max_depth: int = 4
    ) -> Dict[str, Any]:
        # Beam item: (accumulated_steps, cumulative_score)
        beams: List[Tuple[List[str], float]] = [([], 1.0)]

        for depth in range(max_depth):
            candidates: List[Tuple[List[str], float]] = []
            
            for steps, current_score in beams:
                # Generate candidate next steps
                next_step_options = step_generator_fn(problem, steps)
                if not next_step_options:
                    # Trajectory terminated (final answer reached)
                    candidates.append((steps, current_score))
                    continue

                for opt in next_step_options:
                    context = "\n".join(steps)
                    step_score = self.verifier.score_step(context, opt)
                    # Multiplicative trajectory score
                    new_score = current_score * step_score
                    candidates.append((steps + [opt], new_score))

            # Select top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_width]

        best_steps, best_score = beams[0] if beams else ([], 0.0)
        return {
            "best_trajectory": best_steps,
            "score": round(best_score, 4),
            "beams_considered": len(beams)
        }
