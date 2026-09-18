import numpy as np

def explained_variance_ratio(X):
    # Convert input to a concrete numpy array
    X = np.array(X, dtype=np.float64)
    n_samples, n_features = X.shape
    
    # 1. Center the data by subtracting the mean of each column
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Compute the covariance matrix using the unbiased estimator (n - 1)
    # Rowvar=False ensures columns are treated as features, rows as samples
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Extract eigenvalues and eigenvectors
    # np.linalg.eigh is optimized for symmetric/Hermitian matrices like covariance matrices
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvalues in descending order (largest variance first)
    eigenvalues = eigenvalues[::-1]
    
    # 5. Calculate the ratio of each eigenvalue to the total sum
    total_variance = np.sum(eigenvalues)
    ratios = eigenvalues / total_variance
    
    # Return as a clean python list
    return ratios.tolist()