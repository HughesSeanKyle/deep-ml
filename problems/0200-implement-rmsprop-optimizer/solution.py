import numpy as np
def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
    """
    Perform RMSProp optimization update.
    """
    # Convert incoming native Python lists into NumPy arrays for vectorized calculations
    p = np.array(params, dtype=float)
    g = np.array(grads, dtype=float)
    v = np.array(cache, dtype=float)
    
    # Step 1: Update the cache vector using the exponentially decaying average of squared gradients
    new_cache = beta * v + (1.0 - beta) * (g ** 2)
    
    # Step 2: Apply adaptive updates to the parameter weights
    new_params = p - lr * (g / (np.sqrt(new_cache) + epsilon))
    
    # Convert the processed NumPy arrays back into native Python lists for the final output
    return new_params.tolist(), new_cache.tolist()