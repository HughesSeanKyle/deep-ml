import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient with a fully vectorized Adam update rule.
    '''
    # Step 1: Generate a unique structural key for this specific parameter matrix
    param_id = id(param)
    
    # Step 2: Dynamically initialize tracking components if this is the first iteration
    if param_id not in state:
        state[param_id] = {
            'step': 0,
            'm': np.zeros_like(param),  # First moment vector matching shape exactly
            'v': np.zeros_like(param)   # Second moment vector matching shape exactly
        }
        
    # Extract structural tensors from our local state namespace
    param_state = state[param_id]
    param_state['step'] += 1
    
    t = param_state['step']
    m = param_state['m']
    v = param_state['v']
    
    # Hyperparameters for stable Adam convergence
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # Step 3: Compute the raw exponentially decaying moment updates
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)
    
    # Save the updated raw moments back into the state collection
    param_state['m'] = m_new
    param_state['v'] = v_new
    
    # Step 4: Calculate the bias-corrected variants
    m_hat = m_new / (1.0 - (beta1 ** t))
    v_hat = v_new / (1.0 - (beta2 ** t))
    
    # Step 5: Complete the adaptive parameter step
    # Moving in the opposite direction of the gradient to minimize loss
    new_param = param - lr * (m_hat / (np.sqrt(v_hat) + epsilon))
    
    return new_param, state
