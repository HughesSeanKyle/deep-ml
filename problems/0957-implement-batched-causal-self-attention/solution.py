import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value) -> list:
    """
    Compute batched causal self-attention over 3D inputs.

    Args:
        X: Input tensor of shape (B, T, d_in)
        W_query, W_key, W_value: Projection matrices of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) representing the context vectors.
    """
    X = np.asarray(X, dtype=float)
    W_query = np.asarray(W_query, dtype=float)
    W_key = np.asarray(W_key, dtype=float)
    W_value = np.asarray(W_value, dtype=float)
    
    B, T, _ = X.shape
    d_out = W_query.shape[-1]
    
    # 1. Project X into Q, K, V: each has shape (B, T, d_out)
    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value
    
    # 2. Compute scaled dot-product attention scores: shape (B, T, T)
    # Swap only the last two axes of K so batch dimension B stays intact
    K_T = np.swapaxes(K, -1, -2)
    scores = (Q @ K_T) / np.sqrt(d_out)
    
    # 3. Apply causal mask (upper triangle strictly above diagonal -> -inf)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)
    
    # 4. Numerically stable softmax along the last axis
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 5. Aggregate values using attention weights: shape (B, T, d_out)
    context = attn_weights @ V
    
    return context.tolist()
