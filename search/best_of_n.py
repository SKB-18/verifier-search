from typing import Callable, List, Optional, Any

def run_best_of_n(
    prompt: str, 
    n: int, 
    generator_fn: Callable[[str, int], List[str]], 
    verifier_fn: Callable[[str, str], float]
) -> Optional[str]:
    """
    Implements Best-of-N sampling (also known as rejection sampling).
    1. Generates N candidates from a prompt.
    2. Scores each candidate using a verifier function.
    3. Returns the candidate with the highest score.
    """
    candidates = generator_fn(prompt, n)
    
    if not candidates:
        return None
        
    best_candidate = None
    best_score = float('-inf')
    
    for candidate in candidates:
        score = verifier_fn(prompt, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
            
    return best_candidate
