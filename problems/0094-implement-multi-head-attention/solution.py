import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    Supports both 2D matrices and 3D multi-head tensor operations.
    
    Args:
        Q: Query matrix of shape (..., seq_len, d_k)
        K: Key matrix of shape (..., seq_len, d_k)
        V: Value matrix of shape (..., seq_len, d_k)
    
    Returns:
        Attention output of shape (..., seq_len, d_k)
    """
    d_k = Q.shape[-1]
    
    # Compute raw scores using a matrix transpose over the last two axes
    # Transposing axes -2 and -1 swaps seq_len and d_k
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
    
    # Numerically stable softmax row-wise (along the final axis)
    row_maxima = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_maxima)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Weighted aggregation of the value matrix
    output = np.matmul(attention_weights, V)
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    d_head = d_model // n_heads
    
    # 1. Reshape into (seq_len, n_heads, d_head)
    Q_split = Q.reshape(seq_len, n_heads, d_head)
    K_split = K.reshape(seq_len, n_heads, d_head)
    V_split = V_split = V.reshape(seq_len, n_heads, d_head)
    
    # 2. Transpose to (n_heads, seq_len, d_head) to isolate heads
    Q_heads = Q_split.transpose(1, 0, 2)
    K_heads = K_split.transpose(1, 0, 2)
    V_heads = V_split.transpose(1, 0, 2)
    
    # 3. Evaluate parallel scaled dot-product attention
    # Output shape: (n_heads, seq_len, d_head)
    attn_out_heads = self_attention(Q_heads, K_heads, V_heads)
    
    # 4. Transpose back to (seq_len, n_heads, d_head)
    attn_out_split = attn_out_heads.transpose(1, 0, 2)
    
    # 5. Flatten the last two dimensions to recreate (seq_len, d_model)
    output = attn_out_split.reshape(seq_len, d_model)
    
    return output