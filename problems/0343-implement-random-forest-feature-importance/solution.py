import numpy as np
def random_forest_feature_importance(trees: list, n_features: int) -> list:
    # 1. Initialize an absolute accumulator array filled with zeros
    importance_scores = np.zeros(n_features, dtype=float)
    
    # 2. Iterate through all trees and cumulative sub-node dict splits
    for tree in trees:
        for split in tree:
            feat_idx = split['feature_index']
            decrease_val = split['impurity_decrease']
            # Accumulate the reduction value into the matching feature slot
            importance_scores[feat_idx] += decrease_val
            
    total_impurity_decrease = np.sum(importance_scores)
    
    # 3. Handle empty forest configurations or zero-split edge cases safely
    if total_impurity_decrease == 0.0:
        return importance_scores.tolist()
        
    # 4. Normalize the elements so they sum up to 1.0
    normalized_importances = importance_scores / total_impurity_decrease
    
    return normalized_importances.tolist()