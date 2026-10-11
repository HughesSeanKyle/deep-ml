import torch

def softmax(t, dim):
    """
    Computes a numerically stable softmax along the specified dimension.
    """
    # Step 1: Find the maximum values along the target dimension.
    # keepdim=True ensures the tensor keeps its structural axes for safe broadcasting.
    # [0] is accessed because torch.max returns a tuple of (values, indices).
    max_vals = torch.max(t, dim=dim, keepdim=True)[0]
    
    # Step 2: Subtract the max values to stabilize exponents
    shifted_t = t - max_vals
    
    # Step 3: Compute the element-wise exponential
    exp_t = torch.exp(shifted_t)
    
    # Step 4: Sum the exponentials along the chosen axis
    sum_exp = torch.sum(exp_t, dim=dim, keepdim=True)
    
    # Step 5: Normalize to yield valid probabilities
    return exp_t / sum_exp
