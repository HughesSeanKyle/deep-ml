import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    # Ensure inputs are concrete numpy structures
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    
    n_samples, n_features = X.shape
    C = int(np.max(y) + 1)  # Automatically determine number of classes
    
    # 1. Add bias column of ones as the first column of X
    ones_column = np.ones((n_samples, 1), dtype=np.float64)
    X_with_bias = np.hstack((ones_column, X))  # Shape: (n_samples, n_features + 1)
    M = X_with_bias.shape[1]
    
    # 2. Weight Initialization: Initialize all coefficients to zero
    # Shape: (C, M) where each row contains coefficients for one class
    W = np.zeros((C, M), dtype=np.float64)
    
    # 3. One-hot encode the true target labels for matrix multiplication
    Y_onehot = np.zeros((n_samples, C), dtype=np.float64)
    Y_onehot[np.arange(n_samples), y] = 1.0
    
    losses = []
    
    # 4. Core Optimization Loop
    for _ in range(iterations):
        # Calculate raw linear logits: (n_samples, M) @ (M, C) -> (n_samples, C)
        Z = X_with_bias @ W.T
        
        # Safe Softmax evaluation by subtracting max for numerical stability
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        # Compute sum-based Cross Entropy loss (add small epsilon to guard logs)
        epsilon = 1e-15
        loss_val = -np.sum(Y_onehot * np.log(P + epsilon))
        losses.append(round(loss_val, 4))
        
        # Calculate sum-based gradient: (n_samples, C).T @ (n_samples, M) -> (C, M)
        error = P - Y_onehot
        gradient = error.T @ X_with_bias
        
        # Update coefficients
        W = W - learning_rate * gradient
        
    # Format the matrix to round outputs to 4 decimal places for presentation alignment
    coefficients_list = np.round(W, 4).tolist()
    
    return coefficients_list, losses