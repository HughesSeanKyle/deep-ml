import numpy as np

def train(X_train, y_train, X_val, y_val):
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_val = np.array(X_val, dtype=np.float64)
    y_val = np.array(y_val, dtype=np.float64)
    
    n_samples, n_features = X_train.shape
    
    # 1. Prepend bias column of ones to both Train and Validation feature matrices
    X_tr_bias = np.hstack((np.ones((n_samples, 1)), X_train))
    X_val_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
    
    # 2. Build the modified identity penalty matrix (Do NOT penalize the bias at index 0)
    penalty_matrix = np.eye(n_features + 1)
    penalty_matrix[0, 0] = 0.0
    
    # Pre-compute core matrix components to optimize grid execution speed
    XTX = X_tr_bias.T @ X_tr_bias
    XTy = X_tr_bias.T @ y_train
    
    # Validation baseline components for R² calculations
    y_val_mean = np.mean(y_val)
    ss_tot = np.sum((y_val - y_val_mean) ** 2)
    
    best_r2 = float('-inf')
    best_w = None
    
    # 3. Exhaustive search across possible regularization scales
    # Testing log-spaced candidate alphas from highly flexible to highly regularized
    candidate_alphas = np.logspace(-2, 5, 100)
    
    for alpha in candidate_alphas:
        # Normal Equation solver: (XᵀX + αI)⁻¹ Xᵀy
        A = XTX + alpha * penalty_matrix
        
        # Use np.linalg.solve for superior numerical stability over manual inversion (.inv)
        w = np.linalg.solve(A, XTy)
        
        # Evaluate performance on validation split
        val_preds = X_val_bias @ w
        ss_res = np.sum((y_val - val_preds) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        
        # Track the weights that maximize generalization capability
        if r2 > best_r2:
            best_r2 = r2
            best_w = w

    # 4. Construct and return the callable prediction closure capturing the optimized state
    def predict(X):
        X_arr = np.array(X, dtype=np.float64)
        # Re-apply identical bias transformation structure
        X_with_bias = np.hstack((np.ones((X_arr.shape[0], 1)), X_arr))
        return X_with_bias @ best_w
        
    return predict
