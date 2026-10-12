import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    """
    # Step 1: Cast variables to NumPy arrays for seamless vectorized calculation
    p = np.asarray(parameter, dtype=float)
    g = np.asarray(grad, dtype=float)
    m_old = np.asarray(m, dtype=float)
    v_old = np.asarray(v, dtype=float)
    
    # Step 2: Update the first and second moment estimates
    new_m = beta1 * m_old + (1.0 - beta1) * g
    new_v = beta2 * v_old + (1.0 - beta2) * (g ** 2)
    
    # Step 3: Compute the bias correction steps based on the current timestep t
    m_hat = new_m / (1.0 - (beta1 ** t))
    v_hat = new_v / (1.0 - (beta2 ** t))
    
    # Step 4: Execute adaptive parameter update step
    new_parameter = p - learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon))
    
    # Return all updated states rounded to 5 decimal places as required by the format
    return np.round(new_parameter, 5), np.round(new_m, 5), np.round(new_v, 5)