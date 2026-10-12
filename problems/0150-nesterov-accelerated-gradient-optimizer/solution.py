import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    """
    # Step 1: Calculate the look-ahead position
    theta_lookahead = parameter - (momentum * velocity)
    
    # Step 2: Compute the gradient at the future look-ahead position
    g = grad_fn(theta_lookahead)
    
    # Step 3: Update the accumulated velocity tracking term
    new_velocity = (momentum * velocity) + (learning_rate * g)
    
    # Step 4: Perform the parameter step from the original position
    new_parameter = parameter - new_velocity
    
    # Return values rounded to 5 decimal places as required by the template structure
    return np.round(new_parameter, 5), np.round(new_velocity, 5)