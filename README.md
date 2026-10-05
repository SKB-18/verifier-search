# Verifier-Guided Search

This is Repo 3 of the AI Project Portfolio Roadmap.

It implements advanced search algorithms designed to be placed on top of local inference servers (like Repo 1) or reasoning agents (like Repo 2). Instead of relying on single-shot LLM generations, it uses massive parallel generation and verification loops to "search" for the correct answer.

## Implemented Algorithms
- **Best-of-N (Rejection Sampling)**: Generates $N$ candidates in parallel and uses a Verifier model (or programmatic reward function) to select the mathematically highest-scoring response.
- **Monte Carlo Tree Search (MCTS)**: Explores token-level or step-level paths. Uses UCT (Upper Confidence Bound for Trees) to balance exploring new reasoning paths and exploiting known high-reward paths.

## Architecture
- **`search/best_of_n.py`**: Functional implementation of Best-of-N.
- **`search/mcts.py`**: Class-based `MCTSNode` tree state management and UCT selection.

## Testing
Comprehensive TDD tests cover rejection sampling and tree traversal logic.
```bash
python -m pytest tests/
```
