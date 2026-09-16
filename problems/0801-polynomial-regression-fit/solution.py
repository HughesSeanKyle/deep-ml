import numpy as np

def fit_polynomial(x, y, degree):
    # Convert inputs to 1D numpy arrays cleanly
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    
    # 1. Construct the Design Matrix X (Vandermonde Matrix)
    # Generates columns: x^0, x^1, ..., x^degree
    X = np.vander(x_arr, N=degree + 1, increasing=True)
    
    # 2. Set up the Normal Equations pieces
    XT_X = X.T @ X
    XT_y = X.T @ y_arr
    
    # 3. Solve the system (XT_X)c = XT_y directly
    c = np.linalg.solve(XT_X, XT_y)
    
    # Convert back to standard Python list
    return c.tolist()
