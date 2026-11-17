import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    
    # Extract key dimension d_k from the last axis of K
    d_k = K.shape[-1]
    
    # 1. Compute scaled dot-product scores: shape (n_q, n_k)
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # 2. Numerically stable row-wise softmax
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 3. Round to 4 decimal places and convert to nested Python list
    return np.round(weights, 4).tolist()
