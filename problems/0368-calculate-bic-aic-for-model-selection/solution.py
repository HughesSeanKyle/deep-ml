import numpy as np

def calculate_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple:
    # Convert inputs to NumPy arrays just in case lists are passed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n = len(y_true)
    
    # Calculate Residual Sum of Squares (RSS)
    rss = np.sum((y_true - y_pred) ** 2)
    
    # Common error term shared between AIC and BIC equations
    error_term = n * np.log(rss / n)
    
    # Calculate AIC with a constant penalty parameter factor of 2
    aic = error_term + 2 * k
    
    # Calculate BIC with a sample size dependent penalty factor of ln(n)
    bic = error_term + k * np.log(n)
    
    return (round(float(aic), 4), round(float(bic), 4))