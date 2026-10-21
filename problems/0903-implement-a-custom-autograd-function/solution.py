import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """Perform the forward ReLU operation and cache the input tensor."""
        # Save the input tensor for the backward pass
        ctx.save_for_backward(x)
        
        # Apply element-wise max(0, x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate the gradient with respect to the input x using the cached tensor."""
        # Retrieve the saved input tensor
        x, = ctx.saved_tensors
        
        # Create a copy of the upstream gradient to avoid mutating it in-place
        grad_input = grad_output.clone()
        
        # Zero out the gradient elements where the original input was <= 0
        grad_input[x <= 0] = 0
        
        return grad_input
