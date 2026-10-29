import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # 1. Extract dimensionality of the key vectors (d_k is the last dimension)
    d_k = Q.shape[-1]
    
    # 2. Compute the scaled attention scores
    # Shape: (seq_len, seq_len)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # 3. Inject the mask matrix to block future tokens
    # Elements where mask is -inf become -inf
    masked_scores = scores + mask
    
    # 4. Apply numerically stable softmax row-wise
    # Subtracting the maximum values avoids floating-point overflow
    row_maxima = np.max(masked_scores, axis=1, keepdims=True)
    exp_scores = np.exp(masked_scores - row_maxima)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # 5. Aggregate values using the computed attention distribution
    # Shape: (seq_len, d_model)
    output = np.dot(attention_weights, V)
    
    return output