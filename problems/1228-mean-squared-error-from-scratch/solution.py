import torch

def mse(pred, target):
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    # Step 1: Compute element-wise squared differences
    squared_diffs = (pred - target) ** 2
    
    # Step 2: Compute the average across all dimensions
    mean_loss = torch.mean(squared_diffs)
    
    # Step 3: Extract and return the value as a native Python float
    return float(mean_loss.item())
