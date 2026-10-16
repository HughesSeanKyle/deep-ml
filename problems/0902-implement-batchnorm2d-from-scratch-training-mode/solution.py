import torch

def batchnorm2d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Perform training-mode Batch Normalization for a 4D spatial input tensor.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        gamma: Scale parameter tensor of shape (C,)
        beta: Shift parameter tensor of shape (C,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized, scaled, and shifted tensor of shape (N, C, H, W)
    """
    # 1. Compute the mean across the batch (0) and spatial dimensions (2, 3)
    # keepdim=True outputs a shape of (1, C, 1, 1) for seamless broadcasting
    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    
    # 2. Compute the biased (population) variance across the same dimensions
    # correction=0 guarantees a biased calculation (dividing by N*H*W, not N*H*W - 1)
    var = torch.var(x, dim=(0, 2, 3), keepdim=True, correction=0)
    
    # 3. Normalize the input tensor using channel statistics
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # 4. Reshape 1D scale/shift tensors of shape (C,) into (1, C, 1, 1)
    gamma_reshaped = gamma.view(1, -1, 1, 1)
    beta_reshaped = beta.view(1, -1, 1, 1)
    
    # 5. Apply the affine transformation and return
    y = gamma_reshaped * x_norm + beta_reshaped
    
    return y
