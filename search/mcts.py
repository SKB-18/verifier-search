import math
import random
from typing import Callable, List, Optional, Any

class MCTSNode:
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.is_expanded = False

    def uct_score(self, exploration_weight: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

def run_mcts(
    initial_state: Any,
    num_simulations: int,
    generate_fn: Callable[[Any], List[Any]],
    rollout_fn: Callable[[Any], float],
    exploration_weight: float = 1.414
) -> Any:
    """
    Runs Monte Carlo Tree Search to find the best sequence of states.
    """
    root = MCTSNode(state=initial_state)
    
    for _ in range(num_simulations):
        # 1. Selection
        node = root
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda c: c.uct_score(exploration_weight))
            
        # 2. Expansion
        if not node.is_expanded:
            next_states = generate_fn(node.state)
            node.children = [MCTSNode(state=s, parent=node) for s in next_states]
            node.is_expanded = True
            
        # If it's a terminal node with no children, we just evaluate it directly
        eval_node = node
        if node.children:
            # Pick a random child to simulate from (or could use heuristics)
            eval_node = random.choice(node.children)
            
        # 3. Simulation (Rollout)
        reward = rollout_fn(eval_node.state)
        
        # 4. Backpropagation
        curr = eval_node
        while curr is not None:
            curr.visits += 1
            curr.value += reward
            curr = curr.parent

    # Return the best next step from root (most visited)
    if not root.children:
        return initial_state
        
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.state
