
import numpy as np

def gini_impurity(y):
    # 1. Handle edge case for completely empty arrays safely
    if len(y) == 0:
        return 0.0
        
    labels = np.array(y)
    total_samples = len(labels)
    
    # 2. Extract frequencies cleanly using unique value counting
    _, counts = np.unique(labels, return_counts=True)
    
    # 3. Vectorised computation of label probabilities
    probabilities = counts / total_samples
    
    # 4. Gini formula: 1 - sum(p_i^2)
    gini = 1.0 - np.sum(probabilities ** 2)
    
    return float(gini)