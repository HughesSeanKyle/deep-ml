import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    
    Args:
        X: Input tensor of shape (B, T, D)
        gamma: Scale parameter of shape (1, 1, D) or broadcastable to X
        beta: Shift parameter of shape (1, 1, D) or broadcastable to X
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized, scaled, and shifted tensor of shape (B, T, D)
    """
    # 1. Compute mean and variance across the final feature dimension (axis=-1)
    # Output shape: (B, T, 1)
    layer_mean = np.mean(X, axis=-1, keepdims=True)
    layer_var = np.var(X, axis=-1, keepdims=True)
    
    # 2. Normalize the feature dimension per batch and sequence index
    X_norm = (X - layer_mean) / np.sqrt(layer_var + epsilon)
    
    # 3. Apply the broadcastable scale (gamma) and shift (beta)
    out = gamma * X_norm + beta
    
    return out
