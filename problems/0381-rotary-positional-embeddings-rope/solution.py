import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape
    if d % 2 != 0:
        raise ValueError("Embedding dimension d must be even.")
        
    # 1. Compute the inverse frequencies for each pair i = 0, 1, ..., d/2 - 1
    # Formula: theta_i = 1 / (base ^ (2i / d))
    pair_indices = np.arange(0, d // 2, dtype=np.float64)
    freqs = 1.0 / (base ** (2.0 * pair_indices / d))
    
    # 2. Compute rotation angles (m * theta_i) via outer product
    # Shape: (seq_len, d // 2)
    angles = np.outer(positions, freqs)
    
    # 3. Precompute cosine and sine matrices
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    
    # 4. Slice the input tensor into even and odd dimension pairs
    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    
    # 5. Apply the 2D rotation matrix to each pair
    # [x_even'] = [x_even * cos - x_odd * sin]
    # [x_odd' ]   [x_even * sin + x_odd * cos]
    out = np.empty_like(x, dtype=np.float64)
    out[:, 0::2] = x_even * cos_vals - x_odd * sin_vals
    out[:, 1::2] = x_even * sin_vals + x_odd * cos_vals
    
    return out