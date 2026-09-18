import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    # Ensure matrix input format
    X = np.array(X, dtype=np.float64)
    n_samples, n_features = X.shape
    
    # 1. Center the data by subtracting the feature means
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # 2. Compute the population covariance matrix (divide by n)
    cov_matrix = (X_centered.T @ X_centered) / n_samples
    
    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort indices in descending order (largest eigenvalue first)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # 5. Isolate the top n_components eigenvectors
    W = eigenvectors_sorted[:, :n_components]
    
    # 6. Project down to lower-dimensional space (Compression)
    Z = X_centered @ W
    
    # 7. Project back up to original feature space and add the mean (Reconstruction)
    X_reconstructed = (Z @ W.T) + mean
    
    # 8. Calculate Mean Squared Error (MSE)
    mse = np.mean((X - X_reconstructed) ** 2)
    
    return float(mse)