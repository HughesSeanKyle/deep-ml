import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    W = np.asarray(attn_weights, dtype=float)
    T = W.shape[0]
    
    if method == 'tril':
        # 1. Create lower-triangular mask of 1s (including main diagonal)
        mask = np.tril(np.ones((T, T), dtype=float))
        
        # 2. Zero out future positions (upper triangle)
        masked_weights = W * mask
        
        # 3. Re-normalize each row so probabilities sum back to 1.0
        row_sums = masked_weights.sum(axis=-1, keepdims=True)
        result = masked_weights / row_sums
        
    elif method == 'triu':
        # 1. Recover unnormalized log-scores from the strictly positive probabilities
        scores = np.log(W)
        
        # 2. Create upper-triangular boolean mask strictly above diagonal (k=1)
        upper_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        
        # 3. Set future positions to -inf so exp(-inf) = 0
        scores = np.where(upper_mask, -np.inf, scores)
        
        # 4. Numerically stable softmax along the last axis
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        result = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'tril' or 'triu'.")
        
    return result.tolist()
