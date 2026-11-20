import numpy as np

def mha_combined_qkv(x, W_qkv, W_out, num_heads: int) -> list:
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """
    x = np.asarray(x, dtype=float)
    W_qkv = np.asarray(W_qkv, dtype=float)
    W_out = np.asarray(W_out, dtype=float)
    
    batch, num_tokens, _ = x.shape
    d_out = W_out.shape[0]
    head_dim = d_out // num_heads
    
    # 1. Fused QKV projection: (batch, num_tokens, 3 * d_out)
    qkv = x @ W_qkv
    
    # 2. Reshape into (batch, num_tokens, 3, num_heads, head_dim) and split along axis=2
    qkv = qkv.reshape(batch, num_tokens, 3, num_heads, head_dim)
    # Transpose to (3, batch, num_heads, num_tokens, head_dim)
    qkv = np.transpose(qkv, (2, 0, 3, 1, 4))
    Q, K, V = qkv[0], qkv[1], qkv[2]
    
    # 3. Scaled dot-product attention scores: (batch, num_heads, num_tokens, num_tokens)
    K_T = np.swapaxes(K, -1, -2)
    scores = (Q @ K_T) / np.sqrt(head_dim)
    
    # 4. Apply causal mask (upper triangle strictly above diagonal -> -inf)
    causal_mask = np.triu(np.ones((num_tokens, num_tokens), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)
    
    # 5. Numerically stable softmax along the last (key) dimension
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 6. Weighted sum of values: (batch, num_heads, num_tokens, head_dim)
    context = attn_weights @ V
    
    # 7. Concatenate heads back to (batch, num_tokens, d_out)
    context = np.transpose(context, (0, 2, 1, 3)).reshape(batch, num_tokens, d_out)
    
    # 8. Final linear output projection
    out = context @ W_out
    
    return out.tolist()
