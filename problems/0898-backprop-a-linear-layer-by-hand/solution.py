import torch

def linear_backward(grad_output, x, W):
    """
    Computes manual gradients for a linear layer without utilizing autograd.
    
    Returns:
        grad_input (dL/dx), grad_W (dL/dW), grad_b (dL/db)
    """
    # 1. dL/dx = grad_output @ W
    grad_input = grad_output @ W
    
    # 2. dL/dW = grad_output.t() @ x
    grad_W = grad_output.t() @ x
    
    # 3. dL/db = sum of grad_output across the batch dimension (axis 0)
    grad_b = torch.sum(grad_output, dim=0)
    
    return grad_input, grad_W, grad_b
