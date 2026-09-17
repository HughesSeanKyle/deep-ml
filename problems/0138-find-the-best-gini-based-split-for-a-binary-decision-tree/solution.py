import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    n_samples, n_features = X.shape
    
    best_gini = float('inf')
    best_feat_idx = -1
    best_thresh = -1.0
    
    # Helper lambda to compute single node gini quickly
    def _get_gini(labels):
        n = len(labels)
        if n == 0:
            return 0.0
        p1 = np.sum(labels) / n
        p0 = 1.0 - p1
        return 1.0 - (p0**2 + p1**2)

    # 1. Iterate through every feature column
    for feat_idx in range(n_features):
        feature_values = X[:, feat_idx]
        
        # Consider every unique value in the column as a candidate threshold
        unique_thresholds = np.unique(feature_values)
        
        for thresh in unique_thresholds:
            # 2. Split the dataset based on condition
            left_mask = feature_values <= thresh
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            n_l, n_r = len(y_left), len(y_right)
            
            # 3. Calculate separate child impurities
            gini_l = _get_gini(y_left)
            gini_r = _get_gini(y_right)
            
            # 4. Weight the results by sample distributions
            weighted_gini = (n_l / n_samples) * gini_l + (n_r / n_samples) * gini_r
            
            # 5. Track the best result found (strict inequality ensures ties preserve first encounter)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat_idx = feat_idx
                best_thresh = thresh
                
    return best_feat_idx, best_thresh