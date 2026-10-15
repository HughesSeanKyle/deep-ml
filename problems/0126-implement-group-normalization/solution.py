import numpy as np

def group_normalization(
    X: np.ndarray, 
    gamma: np.ndarray, 
    beta: np.ndarray, 
    num_groups: int, 
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: Input tensor of shape (B, C, H, W)
        gamma: Scale parameter of shape (C,) or broadcastable to (1, C, 1, 1)
        beta: Shift parameter of shape (C,) or broadcastable to (1, C, 1, 1)
        num_groups: Number of groups to split the channels into
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized, scaled, and shifted tensor of shape (B, C, H, W)
    """
    B, C, H, W = X.shape
    
    # 1. Reshape tensor to split channels into groups
    # Shape becomes: (B, G, C // G, H, W)
    channels_per_group = C // num_groups
    X_reshaped = X.reshape(B, num_groups, channels_per_group, H, W)
    
    # 2. Compute mean and variance per group across axes (2, 3, 4)
    # Shape of mean and var will be (B, G, 1, 1, 1)
    group_mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    group_var = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    
    # 3. Normalize the reshaped tensor
    X_norm_reshaped = (X_reshaped - group_mean) / np.sqrt(group_var + epsilon)
    
    # 4. Reshape back to the original layout (B, C, H, W)
    X_norm = X_norm_reshaped.reshape(B, C, H, W)
    
    # 5. Ensure gamma and beta are correctly shaped for broadcasting (1, C, 1, 1)
    gamma = np.reshape(gamma, (1, C, 1, 1))
    beta = np.reshape(beta, (1, C, 1, 1))
    
    # 6. Apply per-channel scale and shift parameters
    out = gamma * X_norm + beta
    
    return out
