# --- cell 1 ---
import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate an adversarial example for input x using PGD (Projected Gradient Descent).
    '''
    # Set step size (alpha) and number of iterations for multi-step refinement
    # Typical standard for epsilon=0.1: 10 steps of alpha = epsilon / 4 or 5
    num_steps = 10
    alpha = epsilon / 4.0
    
    # Initialize the adversarial candidate as a copy of the clean image
    x_adv = x.clone().detach()
    
    for _ in range(num_steps):
        # 1. Enable gradient tracking for this step
        x_adv.requires_grad_(True)
        
        # 2. Forward pass
        output = model(x_adv)
        loss = criterion(output, y)
        
        # 3. Backward pass to get local input gradients
        model.zero_grad()
        loss.backward()
        
        # 4. Take a small step in the direction of gradient sign (Gradient Ascent)
        grad_sign = x_adv.grad.sign()
        
        # We detach here to modify the tensor elements outside the autograd graph
        x_adv = x_adv.detach() + alpha * grad_sign
        
        # 5. Projection Step: Clip perturbation to ensure it stays within the L_infinity epsilon ball
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = x + delta
        
        # 6. Clip to ensure valid pixel bounds [0, 1]
        x_adv = torch.clamp(x_adv, min=0.0, max=1.0)
        
    return x_adv.detach()