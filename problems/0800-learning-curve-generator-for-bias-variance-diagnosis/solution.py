import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    # 1. Standardise inputs into 1D numpy feature arrays
    X_tr = np.array(X_train, dtype=float).flatten()
    y_tr = np.array(y_train, dtype=float)
    X_v = np.array(X_val, dtype=float).flatten()
    y_v = np.array(y_val, dtype=float)
    
    train_errors = []
    val_errors = []
    
    # 2. Iterate through each subset step size
    for n in train_sizes:
        # Extract training data prefixes
        X_tr_sub = X_tr[:n]
        y_tr_sub = y_tr[:n]
        
        # Build Polynomial Design Matrices using Vandermonde helper
        X_poly_train = np.vander(X_tr_sub, N=degree + 1, increasing=True)
        X_poly_val = np.vander(X_v, N=degree + 1, increasing=True)
        
        # Compute weights vector using Moore-Penrose Pseudoinverse
        w = np.linalg.pinv(X_poly_train) @ y_tr_sub
        
        # Make predictions
        y_pred_train = X_poly_train @ w
        y_pred_val = X_poly_val @ w
        
        # Calculate Mean Squared Error (MSE)
        train_mse = np.mean((y_pred_train - y_tr_sub) ** 2)
        val_mse = np.mean((y_pred_val - y_v) ** 2)
        
        train_errors.append(train_mse)
        val_errors.append(val_mse)
        
    # 3. Perform Diagnostic Evaluation on the final index elements
    final_train_error = train_errors[-1]
    final_val_error = val_errors[-1]
    
    if final_train_error > bias_threshold:
        diagnosis = 'high_bias'
    elif (final_val_error - final_train_error) > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'
        
    return {
        'train_errors': train_errors,
        'val_errors': val_errors,
        'diagnosis': diagnosis
    }