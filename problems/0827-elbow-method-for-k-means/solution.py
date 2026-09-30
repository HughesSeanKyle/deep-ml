import numpy as np

def elbow_wcss(X: np.ndarray, k_values: list, max_iters: int = 100) -> list:
    # 1. Cast features matrix to a structured NumPy array
    X_arr = np.asarray(X, dtype=float)
    n_samples, n_features = X_arr.shape
    
    wcss_results = []
    
    for k in k_values:
        # 2. Strict Grading Rule: Initialize centroids as the first k rows of X
        centroids = X_arr[:k].copy()
        
        # Track assignments to monitor early convergence stopping
        old_assignments = np.full(n_samples, -1, dtype=int)
        
        for iteration in range(max_iters):
            # 3. Vectorized Assignment Step
            # Compute distances from all samples to all current centroids
            # Shape: (n_samples, k, n_features)
            sq_diffs = (X_arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2
            sq_distances = np.sum(sq_diffs, axis=2)
            
            # Find the index of the closest centroid for each point
            assignments = np.argmin(sq_distances, axis=1)
            
            # Check for early stopping condition
            if np.array_equal(assignments, old_assignments):
                break
            old_assignments = assignments.copy()
            
            # 4. Centroid Update Step
            for j in range(k):
                cluster_mask = (assignments == j)
                if np.any(cluster_mask):
                    centroids[j] = np.mean(X_arr[cluster_mask], axis=0)
                else:
                    # If a cluster is empty, leave its centroid unchanged
                    pass
                    
        # 5. Calculate Final WCSS for the current k configuration
        final_sq_diffs = (X_arr - centroids[assignments]) ** 2
        final_wcss = np.sum(final_sq_diffs)
        
        # Round final float score to 4 decimal places
        wcss_results.append(round(float(final_wcss), 4))
        
    return wcss_results
