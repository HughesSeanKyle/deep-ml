import numpy as np

def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    n_samples, n_features = X_arr.shape
    
    # 1. Initialize all sample labels to -1 (treating everything as noise initially)
    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)
    
    # 2. Vectorized distance calculation to generate the neighborhood lookup matrix
    # Shape: (n_samples, n_samples)
    diffs = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    current_cluster_id = 0
    
    # 3. Outer iteration loop scanning for core points
    for i in range(n_samples):
        if visited[i]:
            continue
            
        visited[i] = True
        
        # Identify all indices sitting inside the current epsilon circle boundaries
        neighbors = np.where(distance_matrix[i] <= eps)[0]
        
        # Check Core Point condition (including the target point itself)
        if len(neighbors) < min_samples:
            # Temporarily leave labeled as noise (-1)
            continue
            
        # 4. Core point discovered: Initialize cluster expansion loop
        labels[i] = current_cluster_id
        
        # Create a dynamic queue list starting with the core point's neighbors
        queue = list(neighbors)
        
        # Use an index pointer to step through the queue dynamically (highly memory efficient)
        queue_idx = 0
        while queue_idx < len(queue):
            neighbor_sample = queue[queue_idx]
            queue_idx += 1
            
            # If the neighbor point was previously classified as noise, it's a border point
            if labels[neighbor_sample] == -1:
                labels[neighbor_sample] = current_cluster_id
                
            if visited[neighbor_sample]:
                continue
                
            visited[neighbor_sample] = True
            labels[neighbor_sample] = current_cluster_id
            
            # Evaluate if this expanded point is also a core point in the dense region
            sub_neighbors = np.where(distance_matrix[neighbor_sample] <= eps)[0]
            if len(sub_neighbors) >= min_samples:
                # Append unseen neighbors to the expansion queue
                for sn in sub_neighbors:
                    queue.append(sn)
                    
        # Increment cluster counter after expanding the current region
        current_cluster_id += 1
        
    return labels