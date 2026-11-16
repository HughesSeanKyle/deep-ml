import numpy as np

def train_bigram_nn(xs: list[int], ys: list[int], vocab_size: int, lr: float = 10.0, num_iters: int = 100, alpha: float = 0.0) -> list[list[float]]:
    """
    Train a one-layer neural-net bigram language model with full-batch gradient descent.

    Args:
        xs: list[int] of length N - input character indices in [0, vocab_size)
        ys: list[int] of length N - target next-character indices in [0, vocab_size)
        vocab_size: int V - number of distinct characters
        lr: float - learning rate
        num_iters: int - number of gradient descent steps
        alpha: float - L2 regularization strength on mean(W**2)

    Returns:
        list[list[float]] - trained weight matrix W of shape (V, V)
    """
    V = vocab_size
    N = len(xs)
    
    # Initialize weight matrix W as zeros of shape (V, V)
    W = np.zeros((V, V), dtype=float)
    
    if num_iters <= 0 or N == 0:
        return W.tolist()
        
    # One-hot encode inputs xs -> X of shape (N, V) and targets ys -> Y of shape (N, V)
    X = np.zeros((N, V), dtype=float)
    X[np.arange(N), xs] = 1.0
    
    Y = np.zeros((N, V), dtype=float)
    Y[np.arange(N), ys] = 1.0
    
    # Full-batch gradient descent loop
    for _ in range(num_iters):
        # 1. Forward pass: compute logits Z = X @ W
        Z = X @ W
        
        # 2. Numerically stable row-wise softmax
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        # 3. Compute analytical gradient: (1/N) * X^T (P - Y) + (2 * alpha / V^2) * W
        grad_data = (X.T @ (P - Y)) / N
        grad_reg = (2.0 * alpha / (V ** 2)) * W
        dW = grad_data + grad_reg
        
        # 4. Update weights
        W -= lr * dW
        
    return W.tolist()