import numpy as np

def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    """
    Update parameters using the Adagrad optimizer.
    """
    # Step 1: Input Validation & Conversion to NumPy arrays
    # Using np.asarray lets us work with arrays, lists, or raw scalars smoothly
    p = np.asarray(parameter, dtype=float)
    g = np.asarray(grad, dtype=float)
    history_G = np.asarray(G, dtype=float)
    
    # Ensure dimensions line up perfectly to prevent stealth broadcasting errors
    if g.shape != p.shape or history_G.shape != p.shape:
        raise ValueError("Shapes of parameter, grad, and G must match exactly.")
        
    # Step 2: Accumulate the current step's squared gradient
    new_G = history_G + (g ** 2)
    
    # Step 3: Compute the adaptive parameter update
    new_parameter = p - learning_rate * (g / (np.sqrt(new_G) + epsilon))
    
    # Return both components rounded to 5 decimal places as required by the template
    return np.round(new_parameter, 5), np.round(new_G, 5)