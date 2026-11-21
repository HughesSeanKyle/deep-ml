import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    X = np.asarray(X, dtype=float)
    batch_size, seq_len, d_model = X.shape
    d_head = d_model // num_heads

    def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)  # Population variance (ddof=0)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta

    # 1. Multi-Head Self-Attention (Bidirectional / Unmasked for Encoder)
    Q = X @ weights['W_q']
    K = X @ weights['W_k']
    V = X @ weights['W_v']

    # Split into heads: (batch_size, seq_len, num_heads, d_head) -> (batch_size, num_heads, seq_len, d_head)
    Q = Q.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    # Scaled dot-product attention scores
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_head)

    # Numerically stable softmax along the key axis
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Aggregate values and concatenate heads back to (batch_size, seq_len, d_model)
    context = (attn_weights @ V).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    attn_out = context @ weights['W_o']

    # 2. First Add & Norm (Post-LN)
    X_norm1 = layer_norm(X + attn_out, weights['gamma1'], weights['beta1'])

    # 3. Position-wise Feed-Forward Network: ReLU(x @ W1 + b1) @ W2 + b2
    ffn_hidden = np.maximum(0.0, X_norm1 @ weights['W1'] + weights['b1'])
    ffn_out = ffn_hidden @ weights['W2'] + weights['b2']

    # 4. Second Add & Norm (Post-LN)
    out = layer_norm(X_norm1 + ffn_out, weights['gamma2'], weights['beta2'])

    return out