import torch

def adam_step(w, grad, m, v, t, lr, beta1, beta2, eps):
    """
    One Adam update with bias-corrected moments using out-of-place torch ops.
    """
    # Step 1: Calculate the updated first and second moment moving averages
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)
    
    # Step 2: Apply the analytical bias corrections using the 1-indexed timestep t
    m_hat = m_new / (1.0 - (beta1 ** t))
    v_hat = v_new / (1.0 - (beta2 ** t))
    
    # Step 3: Perform the adaptive parameter scaling update step
    w_new = w - lr * (m_hat / (torch.sqrt(v_hat) + eps))
    
    return w_new, m_new, v_new
