import numpy as np

def permutation_importance(X, y, weights, bias, permutations):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    
    n_samples, n_features = X.shape
    
    # 1. Helper function to compute R² score
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    
    def compute_r2(preds):
        # Handle edge case where ss_tot is zero to prevent division-by-zero errors
        if ss_tot == 0:
            return 0.0
        ss_res = np.sum((y - preds) ** 2)
        return 1.0 - (ss_res / ss_tot)
        
    # 2. Compute Baseline R²
    baseline_preds = X @ weights + bias
    baseline_r2 = compute_r2(baseline_preds)
    
    importance_scores = []
    
    # 3. Evaluate each feature column
    for j in range(n_features):
        feature_permutations = permutations[j]
        shuffled_r2_scores = []
        
        # Run through each provided shuffle scenario for this column
        for perm in feature_permutations:
            # Create a localized clone of the matrix to avoid cross-contamination
            X_shuffled = X.copy()
            
            # Scramble only the designated column j using the explicit permutation indices
            X_shuffled[:, j] = X[perm, j]
            
            # Predict with the corrupted matrix and score it
            shuffled_preds = X_shuffled @ weights + bias
            shuffled_r2 = compute_r2(shuffled_preds)
            shuffled_r2_scores.append(shuffled_r2)
            
        # 4. Importance is the baseline score MINUS the mean of the degraded scores
        mean_shuffled_r2 = np.mean(shuffled_r2_scores)
        importance = baseline_r2 - mean_shuffled_r2
        importance_scores.append(float(importance))
        
    return importance_scores