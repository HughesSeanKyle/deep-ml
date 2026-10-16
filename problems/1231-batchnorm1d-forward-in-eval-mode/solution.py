import torch

def bn_eval(
    x: torch.Tensor, 
    mean: torch.Tensor, 
    var: torch.Tensor, 
    gamma: torch.Tensor, 
    beta: torch.Tensor, 
    eps: float = 1e-5
) -> torch.Tensor:
    """Apply batch-norm inference normalization using frozen statistics.

    Args:
        x (Tensor): input tensor of any shape
        mean (Tensor): running mean tensor broadcastable against x
        var (Tensor): running variance tensor broadcastable against x
        gamma (Tensor): scale parameter tensor broadcastable against x
        beta (Tensor): shift parameter tensor broadcastable against x
        eps (float): numerical stability constant

    Returns:
        Tensor: normalized and affine-transformed tensor
    """
    # 1. Normalize using the provided running mean and variance statistics
    # PyTorch automatically broadcasts the statistical parameters across matching axes
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # 2. Apply scale (gamma) and shift (beta) transformations
    y = gamma * x_norm + beta
    
    return y
