import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        Q: Query matrix of shape (seq_len_q, d_k)
        K: Key matrix of shape (seq_len_k, d_k)
        V: Value matrix of shape (seq_len_k, d_v)
        mask: Optional binary mask of shape (seq_len_q, seq_len_k)
    
    Returns:
        Tuple of (output, attention_weights)
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    
    # 1. Compute scaled attention scores: shape (seq_len_q, seq_len_k)
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # 2. Apply optional binary mask (0s become -inf so softmax zeroes them out)
    if mask is not None:
        mask = np.asarray(mask)
        scores = np.where(mask == 0, -np.inf, scores)
        
    # 3. Numerically stable row-wise softmax
    max_scores = np.max(scores, axis=-1, keepdims=True)
    # Guard against rows where every position is masked out (-inf - (-inf) = nan)
    max_scores = np.where(np.isinf(max_scores), 0.0, max_scores)
    
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_weights = np.where(sum_exp == 0, 0.0, exp_scores / sum_exp)
    
    # 4. Compute weighted sum of values: shape (seq_len_q, d_v)
    output = attention_weights @ V
    
    return output, attention_weights