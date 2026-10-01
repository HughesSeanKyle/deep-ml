import numpy as np

def gmm_m_step(X, gamma):
    # 1. Safely convert input formats to robust numpy float arrays
    X_arr = np.asarray(X, dtype=float)     # Shape: (N, D)
    gamma_arr = np.asarray(gamma, dtype=float) # Shape: (N, K)
    
    N, D = X_arr.shape
    _, K = gamma_arr.shape
    
    # 2. Compute the effective sample size vector for each cluster component
    # Summing column-wise down rows (axis=0) yields a shape: (K,)
    N_k = np.sum(gamma_arr, axis=0)
    
    # Inject an epsilon buffer to prevent division by zero on empty clusters
    N_k_stable = np.where(N_k == 0, 1e-15, N_k)
    
    # 3. Update Mixing Coefficients (Priors)
    mixing_coeffs = N_k / N
    
    # 4. Vectorized Update for Means
    # gamma_arr.T is (K, N), dot product with X_arr (N, D) yields (K, D)
    # We divide row-wise by the stable cluster sizes vector
    means = np.dot(gamma_arr.T, X_arr) / N_k_stable[:, np.newaxis]
    
    # 5. Vectorized Update for Covariances Matrix Stack
    covariances = np.zeros((K, D, D))
    
    for k in range(K):
        # Calculate coordinate deviations relative to the updated mean vector of cluster k
        # Shape: (N, D)
        diff = X_arr - means[k]
        
        # Pull out responsibility weights column slice for cluster k: shape (N,)
        gamma_k = gamma_arr[:, k]
        
        # Apply weights element-wise to rows via broadcasting: shape (N, D)
        weighted_diff = diff * gamma_k[:, np.newaxis]
        
        # Efficient matrix dot product performs the weighted outer products summation
        # (D, N) dot (N, D) results in the target (D, D) covariance layout matrix
        covariances[k] = np.dot(weighted_diff.T, diff) / N_k_stable[k]
        
    return means, covariances, mixing_coeffs