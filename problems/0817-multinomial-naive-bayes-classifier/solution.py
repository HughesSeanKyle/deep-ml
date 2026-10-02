import numpy as np

def multinomial_naive_bayes(X_train, y_train, X_test, alpha=1.0):
    """
    Multinomial Naive Bayes classifier with Laplace smoothing.
    
    Parameters:
    - X_train: (N, D) array of non-negative feature counts
    - y_train: (N,) array of training class labels
    - X_test: (M, D) array of test feature counts
    - alpha: float, Laplace smoothing parameter
    
    Returns:
    - predictions: (M,) numpy array of predicted class labels
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    
    # Unique classes sorted in ascending order
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_samples, n_features = X_train.shape
    
    log_priors = np.zeros(n_classes)
    log_likelihoods = np.zeros((n_classes, n_features))
    
    # Train: Compute Priors and Conditional Log-Probabilities per class
    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        
        # Log Prior: log(N_c / N)
        log_priors[idx] = np.log(len(X_c) / n_samples)
        
        # Aggregated feature counts for class c
        feature_counts_c = np.sum(X_c, axis=0)
        total_count_c = np.sum(feature_counts_c)
        
        # Laplace-smoothed likelihoods: (T_{c,i} + alpha) / (T_c + alpha * D)
        smoothed_prob = (feature_counts_c + alpha) / (total_count_c + alpha * n_features)
        log_likelihoods[idx, :] = np.log(smoothed_prob)
        
    # Predict: Matrix multiplication for vectorized test predictions
    # Shape: (M, D) @ (D, K) -> (M, K)
    log_posteriors = X_test @ log_likelihoods.T + log_priors
    
    # Map index of max posterior score back to original class label
    predicted_indices = np.argmax(log_posteriors, axis=1)
    
    return classes[predicted_indices]