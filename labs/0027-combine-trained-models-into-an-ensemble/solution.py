import numpy as np
from sklearn.linear_model import LogisticRegression

def train_ensemble(base_models, X_val, y_val):
    """
    Combine K pre-trained base classifiers into an ensemble that predicts class
    labels on new data using a Stacking approach.
    """
    # 1. Generate meta-features for the validation set
    # Gather probability distributions from all base models
    val_probs = np.array([m.predict_proba(X_val) for m in base_models])
    
    # Transpose and reshape: 
    # From (K_models, n_samples, n_classes) -> (n_samples, K_models, n_classes) -> (n_samples, K_models * n_classes)
    # This concatenates the probability outputs side-by-side into a single feature vector per sample
    val_features = val_probs.transpose(1, 0, 2).reshape(X_val.shape[0], -1)
    
    # 2. Train the meta-learner
    # Logistic regression with mild L2 regularization (default C=1.0) is ideal to prevent overfitting
    # on the small 300-sample validation set while learning the models' reliabilities
    meta_learner = LogisticRegression(max_iter=2000)
    meta_learner.fit(val_features, y_val)
    
    # 3. Define the prediction callable for new data
    def predict(X):
        # Generate the exact same meta-features for the incoming test data
        test_probs = np.array([m.predict_proba(X) for m in base_models])
        test_features = test_probs.transpose(1, 0, 2).reshape(X.shape[0], -1)
        
        # Predict final integer class labels using the trained meta-learner
        return meta_learner.predict(test_features)

    return predict