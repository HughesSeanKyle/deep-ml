import numpy as np

def qda_predict(X_train, y_train, X_test):
    """
    Predict class labels for X_test using Quadratic Discriminant Analysis (QDA).
    
    Parameters:
    - X_train: array-like of shape (N, D)
    - y_train: array-like of shape (N,)
    - X_test: array-like of shape (M, D)
    
    Returns:
    - predictions: Python list of predicted class labels (integers/original types)
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    
    N, D = X_train.shape
    classes = np.unique(y_train)
    K = len(classes)
    M = X_test.shape[0]
    
    # Matrix to store discriminant score for each test point under each class
    discriminants = np.zeros((M, K))
    
    for k_idx, c in enumerate(classes):
        # Subset data for class c
        X_c = X_train[y_train == c]
        N_c = len(X_c)
        
        # 1. Compute Prior pi_k
        prior_k = N_c / N
        
        # 2. Compute Mean mu_k
        mu_k = np.mean(X_c, axis=0)
        
        # 3. Compute Covariance Sigma_k (MLE: divide by N_c, ddof=0)
        X_c_centered = X_c - mu_k
        Sigma_k = (X_c_centered.T @ X_c_centered) / N_c
        
        # Compute inverse and log-determinant of Sigma_k
        inv_Sigma_k = np.linalg.inv(Sigma_k)
        sign, logdet_k = np.linalg.slogdet(Sigma_k)
        
        # 4. Compute Discriminant for all test samples vectorized
        # Squared Mahalanobis distance: (x - mu_k)^T * Sigma_k^-1 * (x - mu_k)
        diff = X_test - mu_k
        mahalanobis_sq = np.sum((diff @ inv_Sigma_k) * diff, axis=1)
        
        # delta_k(x) = -0.5 * log|Sigma_k| - 0.5 * Mahalanobis_sq + log(prior)
        discriminants[:, k_idx] = -0.5 * logdet_k - 0.5 * mahalanobis_sq + np.log(prior_k)
        
    # Pick class index with max discriminant score (np.argmax breaks ties by taking first occurrence)
    best_class_indices = np.argmax(discriminants, axis=1)
    
    # Map indices back to original class labels and return as list
    return classes[best_class_indices].tolist()