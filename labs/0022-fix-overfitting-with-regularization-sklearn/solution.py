import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    # Convert input sources to concrete numpy arrays [2]
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # 1. Define a broad, dense spectrum of candidate alpha configurations
    candidate_alphas = np.logspace(-3, 5, 200)
    
    # 2. Instantiate RidgeCV to auto-select the best regularization strength [2]
    # cv=None defaults to highly efficient Leave-One-Out (LOO) cross-validation
    model = RidgeCV(alphas=candidate_alphas, cv=None, scoring='r2')
    
    # 3. Fit the regularized weights and bias intercept term [2]
    model.fit(X_train, y_train)
    
    # Optional troubleshooting diagnostic logging
    # print(f"[Sklearn Diagnostic] Selected Optimal Alpha: {model.alpha_:.4f}")
    
    # 4. Construct the callable prediction closure capturing the trained estimator [2]
    def predict(X):
        X_arr = np.asarray(X, dtype=np.float64)
        # Leverage sklearn's underlying prediction engine to map outputs [2]
        return model.predict(X_arr)
        
    return predict
