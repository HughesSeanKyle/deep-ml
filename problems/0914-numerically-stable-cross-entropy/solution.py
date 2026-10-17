import torch

def cross_entropy(logits, targets):
    # 1. Compute the stable log-normalizer for each row: log(sum(exp(x_j - max))) + max
    # torch.logsumexp handles the max subtraction and addition implicitly under the hood.
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    
    # 2. Extract the unnormalized logit corresponding to the correct target class for each row
    # Using advanced indexing: logits[0, targets[0]], logits[1, targets[1]], ...
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    target_logits = logits[batch_indices, targets]
    
    # 3. Compute per-sample loss: log_sum_exp - target_logit
    per_sample_loss = log_sum_exp - target_logits
    
    # 4. Return the scalar mean across the entire batch
    return per_sample_loss.mean()
