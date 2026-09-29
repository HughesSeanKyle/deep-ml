import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    # 1. Cast incoming structures to structured NumPy arrays
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    preds_arr = np.array(current_predictions, dtype=float)
    
    n_samples, n_features = X_arr.shape
    
    # 2. Compute the current residuals
    residuals = y_arr - preds_arr
    
    best_mse = float('inf')
    best_stump_predictions = np.full(n_samples, np.mean(residuals))
    
    # 3. Search for the optimal feature split threshold maximizing error reduction
    for feat_idx in range(n_features):
        feature_values = X_arr[:, feat_idx]
        unique_vals = np.sort(np.unique(feature_values))
        
        # Calculate the midpoints between consecutive unique feature coordinates
        if len(unique_vals) > 1:
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        else:
            continue  # Fall back to global mean if all column features are identical
            
        for thresh in thresholds:
            left_mask = feature_values <= thresh
            right_mask = ~left_mask
            
            # Extract residual arrays matching the split partitions
            res_l = residuals[left_mask]
            res_r = residuals[right_mask]
            
            # Compute leaf predictions using the arithmetic mean of residuals
            pred_l = np.mean(res_l) if len(res_l) > 0 else 0.0
            pred_r = np.mean(res_r) if len(res_r) > 0 else 0.0
            
            # Construct temporary prediction layout array
            stump_preds = np.where(left_mask, pred_l, pred_r)
            
            # Evaluate Mean Squared Error performance
            mse = np.mean((residuals - stump_preds) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_stump_predictions = stump_preds
                
    # 4. Scale by learning rate and apply update step adjustments
    updated_predictions = preds_arr + learning_rate * best_stump_predictions
    
    # 5. Round final array elements to 4 decimal places and return as a standard list
    return np.round(updated_predictions, 4).tolist()