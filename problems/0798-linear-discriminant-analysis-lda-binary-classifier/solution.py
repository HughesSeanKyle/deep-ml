import numpy as np

def lda_predict(X_train, y_train, X_test):
# Convert inputs to numpy arrays for matrix operations
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    
    # 1. Compute per-class means (mu0 and mu1)
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    
    # 2. Compute within-class scatter matrix Sw
    X0_centered = X0 - mu0
    X1_centered = X1 - mu1
    
    # Sw_c = (X_c - mu_c)^T (X_c - mu_c)
    Sw0 = X0_centered.T @ X0_centered
    Sw1 = X1_centered.T @ X1_centered
    Sw = Sw0 + Sw1
    
    # 3. Solve for the projection vector w
    # w = Sw^-1 (mu1 - mu0)
    Sw_inv = np.linalg.inv(Sw)
    w = Sw_inv @ (mu1 - mu0)
    
    # 4. Set the decision threshold c
    # Midpoint projected onto w
    threshold = np.dot(w, (mu0 + mu1) / 2)
    
    # 5. Predict test point labels
    # Project test points: X_test . w 
    scores = X_test @ w
    
    # Predict 1 if score > threshold, else 0
    predictions = (scores > threshold).astype(int)
    
    # Return as a standard Python list of integers
    return predictions.tolist()
