import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    
    # 1. Compute squared norms of each row for both matrices
    # Shapes: sq_norm1 is (n1, 1), sq_norm2 is (1, n2)
    sq_norm1 = np.sum(X1**2, axis=1, keepdims=True)
    sq_norm2 = np.sum(X2**2, axis=1, keepdims=True).T
    
    # 2. Compute the dot product between all pairs: shape (n1, n2)
    dot_product = np.dot(X1, X2.T)
    
    # 3. Use the algebraic identity to get pairwise squared distances
    sq_distances = sq_norm1 + sq_norm2 - 2 * dot_product
    
    # 4. Compute the RBF kernel matrix and round
    kernel_matrix = np.exp(-gamma * sq_distances)
    
    return np.round(kernel_matrix, 4)