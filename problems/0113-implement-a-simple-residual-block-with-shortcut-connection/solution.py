import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """
    Implements a simple 1D residual block with a shortcut connection.
    
    Args:
        x:  1D input array of shape (features,)
        w1: 2D weight matrix for the first layer
        w2: 2D weight matrix for the second layer
        
    Returns:
        A 1D array representing the final output after the residual block.
    """
    # 1. First layer operation: Matrix multiply followed by a ReLU activation
    # w1 @ x executes the dot product layer mapping
    h1 = np.maximum(0, w1 @ x)
    
    # 2. Second layer operation: Matrix multiply (no activation function here)
    h2 = w2 @ h1
    
    # 3. Add the identity shortcut connection (h2 + x) and apply the final ReLU
    out = np.maximum(0, h2 + x)
    
    return out
