import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Calculate the squared Euclidean distances
    # (N, D) -> (N,) after summing across the feature dimension (dim=-1)
    d_pos = torch.sum((anchor - positive) ** 2, dim=-1)
    d_neg = torch.sum((anchor - negative) ** 2, dim=-1)
    
    # Compute losses for each triplet: max(0, d_pos - d_neg + margin)
    losses = torch.clamp(d_pos - d_neg + margin, min=0.0)
    
    # Return the mean over the batch
    return torch.mean(losses)
