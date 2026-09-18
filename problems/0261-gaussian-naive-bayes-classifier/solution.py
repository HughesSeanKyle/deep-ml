import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    # Unique target classes (e.g., [0, 1])
    classes = np.unique(y_train)
    n_samples, n_features = X_train.shape
    
    # Storage maps for parameters
    priors = {}
    means = {}
    variances = {}
    
    epsilon = 1e-9  # Stability buffer
    
    # === 1. FIT PHASE: Learn from training data ===
    for c in classes:
        # Isolate rows belonging exclusively to class c
        X_c = X_train[y_train == c]
        
        # Calculate Prior: P(C_k)
        priors[c] = X_c.shape[0] / n_samples
        
        # Calculate statistical profiles across all features at once
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.var(X_c, axis=0) + epsilon

    # === 2. PREDICT PHASE: Evaluate test points ===
    predictions = []
    
    for x in X_test:
        class_scores = {}
        
        for c in classes:
            # Start with the log of the prior probability
            log_prior = np.log(priors[c])
            
            # Fetch means and variances for this class
            mu = means[c]
            var = variances[c]
            
            # Vectorised Log-Likelihood calculation for all features of point x
            log_likelihood_elements = (
                - 0.5 * np.log(2 * np.pi) 
                - 0.5 * np.log(var) 
                - ((x - mu) ** 2) / (2 * var)
            )
            
            # Sum up all feature logs + prior log to get total class score
            class_scores[c] = log_prior + np.sum(log_likelihood_elements)
            
        # Select the class index that yielded the highest score
        best_class = max(class_scores, key=class_scores.get)
        predictions.append(best_class)
        
    return np.array(predictions)