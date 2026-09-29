import numpy as np

def train_decision_stump(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Finds the optimal feature, exact threshold value, and branch prediction values
    that minimize the misclassification error rate on the bootstrap fold.
    """
    n_samples, n_features = X.shape
    best_error = float('inf')
    
    # Initialize the structural blueprint dictionary for the optimal stump split rule
    best_stump = {
        'feature_idx': 0,
        'threshold': 0.0,
        'direction': 'standard'
    }
    
    for feat_idx in range(n_features):
        feature_values = X[:, feat_idx]
        # Use exact unique training feature coordinates directly as split benchmarks
        thresholds = np.unique(feature_values)
        
        for thresh in thresholds:
            for direction in ['standard', 'inverted']:
                if direction == 'standard':
                    preds = np.where(feature_values > thresh, 1, 0)
                else:
                    preds = np.where(feature_values <= thresh, 1, 0)
                
                error = np.mean(preds != y)
                
                # Capture the optimal configuration minimizing error metrics
                if error < best_error:
                    best_error = error
                    best_stump['feature_idx'] = feat_idx
                    best_stump['threshold'] = thresh
                    best_stump['direction'] = direction
                    
    return best_stump

def predict_decision_stump(X: np.ndarray, stump: dict) -> np.ndarray:
    """Generates discrete binary class predictions from a single trained exact-split stump."""
    feature_values = X[:, stump['feature_idx']]
    if stump['direction'] == 'standard':
        return np.where(feature_values > stump['threshold'], 1, 0)
    return np.where(feature_values <= stump['threshold'], 1, 0)

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using exact-split decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test of shape (n_test_samples,)
    """
    # 1. Standardize types to absolute numpy arrays
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    
    n_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    
    # 2. Instantiate the modern NumPy Generator object with the given seed
    rng = np.random.default_rng(seed)
    
    # Setup matrix to track evaluations across all stumps
    # Shape layout: (n_estimators, n_test_samples)
    all_predictions = np.zeros((n_estimators, n_test_samples), dtype=int)
    
    for i in range(n_estimators):
        # Generate row indices with replacement matching original training fold limits
        bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        
        # Train decision stump mapping using original raw point coordinates
        stump = train_decision_stump(X_bootstrap, y_bootstrap)
        
        # Cache current estimator prediction output row slice
        all_predictions[i, :] = predict_decision_stump(X_test, stump)
        
    # 3. Process ensemble aggregate majority consensus using mean voting threshold rules
    # Mean across columns (axis=0) gives the exact proportion of estimators voting for class 1
    vote_means = np.mean(all_predictions, axis=0)
    
    # Apply strict mathematically specified rule (ties or greater result in class 1)
    final_predictions = (vote_means >= 0.5).astype(int)
        
    return final_predictions