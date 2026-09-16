import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    bias = 0.0
    
    for _ in range(max_iter):
        w_old = w.copy()
        bias_old = bias
        
        # Forward Pass
        y_pred = X @ w + bias
        error = y_pred - y
        
        # 1. Base gradient matches 1/(2N) * MSE derivative
        dw_base = (X.T @ error) / n_samples
        db = np.mean(error)
        
        # 2. Regularisation Gradients matching standard objective formulation
        dw_l1 = alpha1 * np.sign(w)
        dw_l2 = 2 * alpha2 * w
        
        dw = dw_base + dw_l1 + dw_l2
        
        # Step updates
        w -= learning_rate * dw
        bias -= learning_rate * db
        
        # Check convergence
        w_delta = np.sqrt(np.sum((w - w_old) ** 2))
        b_delta = np.abs(bias - bias_old)
        
        if np.sum(np.abs(dw)) < tol:
            break
            
    return w, bias