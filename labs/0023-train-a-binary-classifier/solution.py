import numpy as np
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # 1. Extract shape dimensions
    n_samples, n_features = X_train.shape
    
    # 2. Initialize weights and bias strictly deterministically
    # Using a fixed seed via a local generator for total NumPy determinism
    rng = np.random.default_rng(42)
    w = rng.normal(loc=0.0, scale=0.01, size=n_features)
    b = 0.0
    
    # 3. Hyperparameters for standard gradient descent on standardized features
    learning_rate = 0.5
    epochs = 500
    
    # 4. Vectorized Gradient Descent Optimization Loop
    for _ in range(epochs):
        # Forward pass: compute linear combination and map through sigmoid
        z = X_train @ w + b
        # Vectorized sigmoid function stable protection
        probs = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        
        # Calculate residuals error
        error = probs - y_train
        
        # Compute exact vector gradients for weights and bias
        dw = (X_train.T @ error) / n_samples
        db = np.mean(error)
        
        # Adjust hyperplanes simultaneously
        w -= learning_rate * dw
        b -= learning_rate * db
        
    # 5. Build the callable inference closure trapping the optimized weights
    def predict(X):
        """
        Classifies new samples cleanly into 0s and 1s using the trained weights.
        """
        linear_out = X @ w + b
        # Binary threshold step function at p = 0.5 (maps directly to z >= 0)
        predictions = (linear_out >= 0.0).astype(np.int64)
        return predictions
        
    return predict
