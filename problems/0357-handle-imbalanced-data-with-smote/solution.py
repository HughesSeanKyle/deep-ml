import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    X_minority = np.array(X_minority, dtype=np.float64)
    n_samples, n_features = X_minority.shape
    
    # 1. Edge Case Handling
    k_actual = min(k, n_samples - 1)
    if k_actual == 0 or n_synthetic == 0:
        return np.empty((0, n_features), dtype=np.float64)
        
    synthetic_samples = []
    
    # 2. Sequential Generator Loop
    for _ in range(n_synthetic):
        # Draw 1: Select the base sample index
        i = np.random.randint(0, n_samples)
        x_i = X_minority[i]
        
        # Calculate Euclidean distances from x_i to all minority samples
        distances = np.linalg.norm(X_minority - x_i, axis=1)
        
        # Sort indices by ascending distance
        sorted_indices = np.argsort(distances)
        
        # Exclude x_i itself (which has distance 0.0 at index 0)
        # Keep the top k_actual indices
        neighbor_indices = sorted_indices[1:k_actual + 1]
        
        # Draw 2: Select which neighbor index to use out of the pool
        j = np.random.randint(0, k_actual)
        nn_idx = neighbor_indices[j]
        x_nn = X_minority[nn_idx]
        
        # Draw 3: Generate the random interpolation factor
        gap = np.random.random()
        
        # Interpolate the new point along the vector segment
        x_synthetic = x_i + gap * (x_nn - x_i)
        synthetic_samples.append(x_synthetic)
        
    return np.array(synthetic_samples, dtype=np.float64)