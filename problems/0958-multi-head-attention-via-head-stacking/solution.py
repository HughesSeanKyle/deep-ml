import numpy as np

def multi_head_attention_stack(x, heads) -> np.ndarray:
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    x = np.asarray(x, dtype=float)
    B, T, _ = x.shape
    
    # Precompute the upper-triangular causal mask (strictly above diagonal -> True)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    
    head_outputs = []
    
    for W_q, W_k, W_v in heads:
        W_q = np.asarray(W_q, dtype=float)
        W_k = np.asarray(W_k, dtype=float)
        W_v = np.asarray(W_v, dtype=float)
        
        d_head = W_q.shape[-1]
        
        # 1. Project x into Q, K, V: each of shape (B, T, d_head)
        Q = x @ W_q
        K = x @ W_k
        V = x @ W_v
        
        # 2. Compute scaled dot-product attention scores: shape (B, T, T)
        K_T = np.swapaxes(K, -1, -2)
        scores = (Q @ K_T) / np.sqrt(d_head)
        
        # 3. Apply causal mask before softmax
        scores = np.where(causal_mask, -np.inf, scores)
        
        # 4. Numerically stable softmax over the key axis (last axis)
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 5. Compute per-head context vectors: shape (B, T, d_head)
        context = attn_weights @ V
        head_outputs.append(context)
        
    # Concatenate all H head context tensors along the embedding dimension (last axis)
    return np.concatenate(head_outputs, axis=-1)
