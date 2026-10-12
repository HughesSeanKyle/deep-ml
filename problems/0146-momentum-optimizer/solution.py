import numpy as np

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the momentum optimizer.
    Handles both single scalars and structural multi-dimensional arrays natively.
    """
    # Step 1: Accumulate momentum by calculating the updated velocity state
    new_velocity = (momentum * velocity) + (learning_rate * grad)
    
    # Step 2: Update parameters by stepping in the direction of the velocity vector
    new_parameter = parameter - new_velocity
    
    # Return both states rounded to 5 decimal places as required by the template
    return np.round(new_parameter, 5), np.round(new_velocity, 5)