import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculate the element-wise margin element: 1 - y_true * y_pred
    margins = 1 - (y_true * y_pred)
    
    # Apply max(0, margin) element-wise
    losses = np.maximum(0, margins)
    
    # Calculate the mean and round to 4 decimal places
    average_loss = np.mean(losses)
    
    return float(round(average_loss, 4))