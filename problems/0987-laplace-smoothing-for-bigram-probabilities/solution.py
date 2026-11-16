import numpy as np

def smooth_bigram_probs(N, k: float) -> list[list[float]]:
    """
    N: 2D list or array of bigram counts (V x V)
    k: float, add-k smoothing constant (k >= 0)
    Returns: 2D list of smoothed bigram probabilities (V x V).
    """
    # Convert input counts to a floating-point NumPy array and add smoothing constant k
    smoothed_counts = np.asarray(N, dtype=float) + k
    
    # Sum across columns (axis=1) while keeping 2D shape (V, 1) for clean broadcasting
    row_sums = smoothed_counts.sum(axis=1, keepdims=True)
    
    # Normalize each row so its probabilities sum to 1.0
    P = smoothed_counts / row_sums
    
    return P.tolist()
