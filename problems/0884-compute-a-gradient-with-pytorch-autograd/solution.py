import torch

def grad_of_quadratic(x_value: float) -> float:
    """
    Computes the gradient of f(x) = x^2 + 3x + 2 at x_value using PyTorch autograd.
    """
    # Step 1: Create a float tensor from the input value.
    # We must explicitly set requires_grad=True to build the tracking graph.
    x = torch.tensor(x_value, dtype=torch.float32, requires_grad=True)
    
    # Step 2: Forward pass. 
    # PyTorch implicitly records this entire calculation.
    y = x**2 + 3*x + 2
    
    # Step 3: Backward pass.
    # This runs backpropagation from the scalar output 'y' back to 'x'.
    y.backward()
    
    # Step 4: Extract the gradient from the '.grad' property of our leaf node.
    # We use .item() to extract the raw numerical value as a native Python float.
    return float(x.grad.item())
