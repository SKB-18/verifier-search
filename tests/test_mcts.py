import pytest
from search.mcts import MCTSNode, run_mcts

def test_mcts_basic():
    # Mock generator: from any state, return 2 possible next steps
    def mock_generate_next_steps(state):
        if len(state) >= 3: # max depth 3
            return []
        return [state + "a", state + "b"]
        
    # Mock rollout: simulate to end and score
    def mock_rollout(state):
        # Prefer paths that end with 'b'
        score = state.count("b") * 0.5
        return score
        
    best_path = run_mcts(
        initial_state="",
        num_simulations=10,
        generate_fn=mock_generate_next_steps,
        rollout_fn=mock_rollout
    )
    
    # After 10 simulations, MCTS should heavily favor the 'b' branch
    # If initial state is "", best path is "bbb" theoretically, but the root selects the best next step
    assert "b" in best_path
