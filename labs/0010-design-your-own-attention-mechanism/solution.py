# --- cell 1 ---
import numpy as np

def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    """
    # Extract the feature dimensionality (d_k)
    dim = Q.shape[-1]
    
    # 1. Compute compatibility scores: (batch_size, query_len, key_len)
    # Transpose K along the last two dimensions to perform a batched matrix multiplication
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    
    # 2. Scale scores by the square root of the dimension to avoid vanishing gradients
    scores = scores / np.sqrt(dim)
    
    # 3. Stable Softmax over the key dimension (last axis) to get attention weights
    # Subtract the maximum value per row to prevent numerical overflow (NaNs)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 4. Compute the weighted combination of the values: (batch_size, query_len, dim)
    output = np.matmul(weights, V)
    
    return output