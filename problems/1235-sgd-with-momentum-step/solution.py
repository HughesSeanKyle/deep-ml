import torch

def momentum_step(w, grad, v, lr, mu):
    """
    Performs one out-of-place SGD-with-momentum step using PyTorch tensor operations.
    """
    # Step 1: Calculate the updated velocity tracking tensor
    v_new = mu * v + grad
    
    # Step 2: Calculate the updated parameter weights tensor
    w_new = w - lr * v_new
    
    return w_new, v_new
