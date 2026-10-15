import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    # 1. Compute mean and variance across spatial dimensions (H, W)
    # Output shape: (B, C, 1, 1)
    ins_mean = np.mean(X, axis=(2, 3), keepdims=True)
    ins_var = np.var(X, axis=(2, 3), keepdims=True)
    
    # 2. Normalize the spatial dimensions per batch element and channel
    X_norm = (X - ins_mean) / np.sqrt(ins_var + epsilon)
    
    # 3. Reshape gamma and beta to (1, C, 1, 1) for broadcasting over (B, H, W)
    gamma_reshaped = np.reshape(gamma, (1, X.shape[1], 1, 1))
    beta_reshaped = np.reshape(beta, (1, X.shape[1], 1, 1))
    
    # 4. Scale and shift
    out = gamma_reshaped * X_norm + beta_reshaped
    
    return out
