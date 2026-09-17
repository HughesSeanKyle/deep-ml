import numpy as np

def bias_variance_decomp(predictions, y_true):
    # 1. Cast inputs to formal NumPy arrays
    preds = np.array(predictions, dtype=float)  # Shape: (B, M)
    y_t = np.array(y_true, dtype=float)         # Shape: (M,)
    
    # 2. Compute the mean prediction across models (axis 0) for each test point
    mean_preds = np.mean(preds, axis=0)         # Shape: (M,)
    
    # 3. Calculate metrics per test point
    # Squared Bias: Difference between mean prediction and true value
    bias_sq_per_point = (mean_preds - y_t) ** 2
    
    # Variance: Average squared deviation of predictions from their mean prediction
    # Broadcasting allows 'preds' (B, M) to subtract 'mean_preds' (M,) seamlessly
    variance_per_point = np.mean((preds - mean_preds) ** 2, axis=0)
    
    # MSE: Average squared deviation of predictions from the absolute true target
    mse_per_point = np.mean((preds - y_t) ** 2, axis=0)
    
    # 4. Average across all M test points to find global empirical values
    return {
        'bias_squared': float(np.mean(bias_sq_per_point)),
        'variance': float(np.mean(variance_per_point)),
        'mse': float(np.mean(mse_per_point))
    }
