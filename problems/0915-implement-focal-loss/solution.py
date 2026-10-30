import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0):
    """
    Computes the multi-class focal loss over a batch.
    
    Args:
        logits: Tensor of shape (N, C) containing unnormalised model outputs.
        targets: LongTensor of shape (N,) containing ground-truth class indices.
        gamma: Float focusing parameter to down-weight easy examples.
        
    Returns:
        Scalar tensor representing the mean focal loss over the batch.
    """
    # 1. Compute stable log-probabilities: shape (N, C)
    log_p = F.log_softmax(logits, dim=-1)
    
    # 2. Gather the log-probabilities for the true target classes: shape (N,)
    # This represents log(p_t)
    log_p_t = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # 3. Exponentiate to get the actual probabilities for the true class (p_t)
    p_t = torch.exp(log_p_t)
    
    # 4. Compute the focal modulating/focusing factor: (1 - p_t)^gamma
    focusing_factor = (1.0 - p_t) ** gamma
    
    # 5. Multiply the focusing factor by the standard cross-entropy loss (-log(p_t))
    per_example_loss = -focusing_factor * log_p_t
    
    # 6. Return the mean loss across the batch
    return per_example_loss.mean()
