import torch

def relu(t: torch.Tensor) -> torch.Tensor:
    """
    Element-wise ReLU: max(0, t) using pure torch ops.
    """
    # torch.clamp handles this instantly by enforcing a floor of 0.0
    return torch.clamp(t, min=0.0)

def leaky_relu(t: torch.Tensor, slope: float = 0.01) -> torch.Tensor:
    """
    Element-wise Leaky ReLU with given negative slope using pure torch ops.
    """
    # If the element is greater than 0, keep t. 
    # Otherwise, scale t by the slope parameter.
    return torch.where(t > 0.0, t, slope * t)
