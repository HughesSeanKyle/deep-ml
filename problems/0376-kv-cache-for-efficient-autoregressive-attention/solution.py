import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    # 1. Project the new token into query, key, and value vectors
    q = x_new @ W_Q  # Shape: (d_k,)
    k = x_new @ W_K  # Shape: (d_k,)
    v = x_new @ W_V  # Shape: (d_v,)
    
    # 2. Initialize or append to the KV cache
    # Reshape k and v to 2D rows: (1, d_k) and (1, d_v)
    k_row = k.reshape(1, -1)
    v_row = v.reshape(1, -1)
    
    if cache is None:
        K_new = k_row
        V_new = v_row
    else:
        K_cache, V_cache = cache
        K_new = np.vstack([K_cache, k_row])
        V_new = np.vstack([V_cache, v_row])
        
    # 3. Compute scaled dot-product attention scores over all cached keys
    d_k = q.shape[0]
    scores = (K_new @ q) / np.sqrt(d_k)  # Shape: (seq_len,)
    
    # 4. Apply numerically stable softmax
    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores)
    attn_weights = exp_scores / np.sum(exp_scores)  # Shape: (seq_len,)
    
    # 5. Compute weighted sum of cached values
    output = attn_weights @ V_new  # Shape: (d_v,)
    
    return output, (K_new, V_new)