import torch
import torch.nn as nn

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """
    Constructs a 2-layer multi-layer perceptron (MLP) using nn.Sequential.
    Architecture: Linear -> ReLU -> Linear (Raw Logits Output).
    """
    # Passing child submodules sequentially into the structural container
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    )
