import numpy as np

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    X_arr = np.asarray(X, dtype=float)
    
    if X_arr.size == 0:
        return []
        
    # 1. Compute pairwise dot-product attention scores: shape (T, T)
    scores = X_arr @ X_arr.T
    
    # 2. Apply numerically stable row-wise softmax to get attention weights: shape (T, T)
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 3. Compute context vectors as weighted sums of input embeddings: shape (T, d)
    context_vectors = attn_weights @ X_arr
    
    return context_vectors.tolist()
