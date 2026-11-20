import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float = 0.1, seed: int = 42) -> list[float]:
    """
    Implement a position-wise feed-forward block with residual and dropout.

    Args:
        x: input vector
        W1, b1: first linear layer parameters
        W2, b2: second linear layer parameters
        dropout_p: dropout probability
        seed: random seed for reproducibility

    Returns:
        Output vector after FFN block (rounded to 4 decimals)
    """
    x_arr = np.asarray(x, dtype=float)
    W1_arr = np.asarray(W1, dtype=float)
    b1_arr = np.asarray(b1, dtype=float)
    W2_arr = np.asarray(W2, dtype=float)
    b2_arr = np.asarray(b2, dtype=float)
    
    # 1. First linear transformation + ReLU activation: h = ReLU(W1 @ x + b1)
    h = np.maximum(0.0, W1_arr @ x_arr + b1_arr)
    
    # 2. Second linear transformation: y = W2 @ h + b2
    y = W2_arr @ h + b2_arr
    
    # 3. Apply inverted dropout to y
    np.random.seed(seed)
    draws = np.random.rand(*y.shape)
    if dropout_p > 0.0:
        mask = draws > dropout_p
        y = np.where(mask, y / (1.0 - dropout_p), 0.0)
        
    # 4. Residual connection: add original input x back to y
    out = y + x_arr
    
    return np.round(out, 4).tolist()