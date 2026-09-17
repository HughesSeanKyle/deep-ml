import numpy as np

def k_nearest_neighbors(points, query_point, k):
    # 1. Cast inputs to NumPy arrays for fast vector operations
    pts_arr = np.array(points, dtype=float)
    q_arr = np.array(query_point, dtype=float)
    
    # 2. Vectorised calculation of Euclidean distances
    # Subtracting q_arr (shape (n,)) from pts_arr (shape (m, n)) broadcasts automatically
    squared_diffs = (pts_arr - q_arr) ** 2
    distances = np.sqrt(np.sum(squared_diffs, axis=1))
    
    # 3. Perform a stable sort to extract sorted positional indices
    # kind='stable' preserves input order for identical values
    sorted_indices = np.argsort(distances, kind='stable')
    
    # 4. Extract the top k points using the sorted index masks
    top_k_indices = sorted_indices[:k]
    
    # Convert chosen array records back to native tuples
    return [points[idx] for idx in top_k_indices]