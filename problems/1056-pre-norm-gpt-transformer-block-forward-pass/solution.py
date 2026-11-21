import numpy as np

def pre_norm_transformer_block(x: np.ndarray, params: dict, num_heads: int) -> np.ndarray:
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """
    x = np.asarray(x, dtype=float)
    batch, seq_len, emb_dim = x.shape
    d_head = emb_dim // num_heads
    eps = 1e-5

    def layer_norm(inp: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(inp, axis=-1, keepdims=True)
        var = np.var(inp, axis=-1, keepdims=True)
        return gamma * ((inp - mean) / np.sqrt(var + eps)) + beta

    def gelu(z: np.ndarray) -> np.ndarray:
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))

    # 1. Pre-Norm + Causal Multi-Head Self-Attention + Residual
    h = layer_norm(x, params['ln1_gamma'], params['ln1_beta'])

    Q = (h @ params['W_q']).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K = (h @ params['W_k']).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V = (h @ params['W_v']).reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_head)
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)

    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = (attn_weights @ V).transpose(0, 2, 1, 3).reshape(batch, seq_len, emb_dim)
    h = context @ params['W_o']
    x = x + h

    # 2. Pre-Norm + GELU Feed-Forward Network + Residual
    h = layer_norm(x, params['ln2_gamma'], params['ln2_beta'])
    h = gelu(h @ params['W_ff1'] + params['b_ff1']) @ params['W_ff2'] + params['b_ff2']
    x = x + h

    return x
