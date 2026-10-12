import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step with decoupled weight decay.
    """
    # Step 1: Update the first and second moment moving averages
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g ** 2)
    
    # Step 2: Compute the bias correction steps for both moments
    m_hat = m_new / (1.0 - (beta1 ** t))
    v_hat = v_new / (1.0 - (beta2 ** t))
    
    # Step 3: Apply the decoupled weight decay step
    # and the adaptive gradient step in a single operation
    w_new = w - lr * (m_hat / (np.sqrt(v_hat) + epsilon)) - lr * weight_decay * w
    
    return w_new, m_new, v_new