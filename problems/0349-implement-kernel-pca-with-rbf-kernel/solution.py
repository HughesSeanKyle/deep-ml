import numpy as np

def kernel_pca_rbf(X, n_components, gamma=1.0):
    """
    Perform Kernel PCA using RBF kernel.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, _ = X.shape
    
    # 1. Compute Pairwise Squared Distances efficiently
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = np.sum(X**2, axis=1).reshape(-1, 1)
    dist_matrix = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
    
    # Clip small negative values caused by floating point errors
    dist_matrix = np.maximum(dist_matrix, 0.0)
    
    # 2. Compute RBF Kernel Matrix
    K = np.exp(-gamma * dist_matrix)
    
    # 3. Center the Kernel Matrix
    # K_centered = K - 1_n K - K 1_n + 1_n K 1_n
    ones = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - np.dot(ones, K) - np.dot(K, ones) + np.dot(np.dot(ones, K), ones)
    
    # 4. Eigendecomposition
    # Use eigh for symmetric matrices (faster/more stable than eig)
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components first
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    
    # Filter out/clip negative or near-zero eigenvalues to avoid division by zero or negative sqrt
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # 5. Enforce Sign Convention
    # For each column, find index of max abs value. If negative, flip column.
    for i in range(eigenvectors.shape[1]):
        max_abs_idx = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[max_abs_idx, i] < 0:
            eigenvectors[:, i] *= -1
            
    # 6. Project Data using K_centered and dividing by sqrt(lambda)
    # Formula: Z = K_centered * V * diag(1 / sqrt(lambda))
    Z = np.dot(K_centered, eigenvectors) / np.sqrt(eigenvalues)
    
    return Z