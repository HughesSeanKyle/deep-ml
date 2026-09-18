import numpy as np
from itertools import product

def grid_search(X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                param_grid: dict, model_fn: callable, 
                scoring_fn: callable) -> tuple:
    # 1. Extract keys and value lists to guarantee matching alignment order
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    
    best_score = float('-inf')
    best_params = None
    
    # 2. Generate the Cartesian product across all hyperparameter values
    # itertools.product handles dynamic nested loops cleanly
    for combination in product(*value_lists):
        # Reconstruct the current configuration dictionary
        current_params = dict(zip(keys, combination))
        
        # 3. Train and predict with the model function unpacking the kwargs
        y_pred = model_fn(X_train, y_train, X_val, **current_params)
        
        # 4. Evaluate using the provided scoring function
        current_score = scoring_fn(y_val, y_pred)
        
        # 5. Track the best score
        # Using strict inequality (>) guarantees we keep the first configuration 
        # encountered in case of a tie score, per the requirements.
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            
    # Return the best combination and score rounded to 4 decimal places
    return best_params, round(best_score, 4)