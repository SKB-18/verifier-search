def test_best_of_n_basic():
    from search.best_of_n import run_best_of_n
    
    # Mock generator that returns predictable candidates
    def mock_generate(prompt, n):
        return [f"Candidate {i}" for i in range(n)]
        
    # Mock verifier that scores Candidate 2 the highest
    def mock_verify(prompt, candidate):
        if "2" in candidate:
            return 0.9
        return 0.1
        
    best = run_best_of_n("Solve this math problem", n=5, generator_fn=mock_generate, verifier_fn=mock_verify)
    assert best == "Candidate 2"

def test_best_of_n_empty():
    from search.best_of_n import run_best_of_n
    
    def mock_generate(prompt, n):
        return []
        
    def mock_verify(prompt, candidate):
        return 1.0
        
    best = run_best_of_n("Prompt", n=3, generator_fn=mock_generate, verifier_fn=mock_verify)
    assert best is None
