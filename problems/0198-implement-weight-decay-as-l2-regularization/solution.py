import numpy as np

def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
    updated_parameters = []
    
    # Process each parameter group sequentially
    for i in range(len(parameters)):
        # Convert lists to numpy arrays for efficient vectorized math
        param = np.array(parameters[i], dtype=float)
        grad = np.array(gradients[i], dtype=float)
        
        if apply_to_all[i]:
            # Apply Weight Decay + Gradient Step
            # Formula: w = w * (1 - lr * decay) - lr * grad
            param = param * (1 - lr * weight_decay) - lr * grad
        else:
            # Apply standard Gradient Descent only (No decay for biases)
            param = param - lr * grad
            
        # Convert back to list format to match expected output structure
        updated_parameters.append(np.round(param, 3).tolist())
        
    return updated_parameters