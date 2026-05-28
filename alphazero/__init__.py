"""AlphaZero-style two-player learning system for the bin-clearing puzzle.

Two asymmetric players self-play to co-evolve:
  - Stacker: places obstacles on a discrete grid at episode start to maximize
    solver failure (adversarial puzzle designer).
  - Solver: pushes blocks to clear the target out of the bin's south exit.

Both players run PUCT MCTS guided by their own (policy, value) network.
"""
