import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
# Cast to numpy arrays to enable vectorized operations
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Compute the absolute error for all elements
    error = np.abs(y_true - y_pred)
    
    # Calculate both conditions simultaneously
    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * (error - 0.5 * delta)
    
    # Use np.where to apply the threshold condition element-wise
    loss = np.where(error <= delta, squared_loss, linear_loss)
    
    # Return the average loss cast to a standard Python float
    return float(np.mean(loss))