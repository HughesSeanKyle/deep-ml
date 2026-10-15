import torch

def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization over the last dimension of an arbitrary shaped tensor.
    
    Args:
        x: Input tensor of shape (..., D)
        gamma: Scale parameter tensor of shape (D,)
        beta: Shift parameter tensor of shape (D,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor of shape (..., D)
    """
    # 1. Calculate mean across the final dimension (dim=-1)
    # keepdim=True ensures shape (..., 1) for broadcasting
    mean = torch.mean(x, dim=-1, keepdim=True)
    
    # 2. Calculate population variance across the final dimension (unbiased=False)
    # keepdim=True ensures shape (..., 1) for broadcasting
    var = torch.var(x, dim=-1, keepdim=True, correction=0)  # correction=0 means biased variance
    
    # 3. Normalize the tensor
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # 4. Apply scale (gamma) and shift (beta) via PyTorch broadcasting rules
    y = gamma * x_norm + beta
    
    return y
