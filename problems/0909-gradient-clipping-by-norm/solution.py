import torch
import math
from typing import Iterable

def clip_grad_norm(parameters: Iterable[torch.Tensor], max_norm: float) -> float:
    # 1. Gather all non-None gradients and compute their squared L2 norms
    grad_norms = []
    grads_to_scale = []
    
    for p in parameters:
        if p.grad is not None:
            # detach() ensures this math doesn't track history for backpropagation
            g = p.grad.detach()
            grads_to_scale.append(g)
            
            # Compute sum of squares for this specific gradient tensor
            # (equivalent to squaring the L2 norm)
            grad_norms.append(g.norm(2) ** 2)
            
    # If no parameters have gradients, return 0.0
    if len(grads_to_scale) == 0:
        return 0.0
        
    # 2. Compute the global total L2 norm
    total_norm = float(torch.sqrt(sum(grad_norms)))
    
    # 3. Scale in-place if total norm crosses the maximum budget threshold
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6) # Added small epsilon to prevent division by zero
        for g in grads_to_scale:
            g.mul_(clip_coef) # Modifies gradient values in-place
            
    # 4. Return the unclipped total norm
    return total_norm
