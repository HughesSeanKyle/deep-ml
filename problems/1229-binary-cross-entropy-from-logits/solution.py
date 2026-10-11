import torch

def bce_with_logits(logits, targets):
    """
    Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.
    """
    # Step 1: Compute max(x, 0) element-wise using torch.clamp
    max_val = torch.clamp(logits, min=0.0)
    
    # Step 2: Compute the absolute value of the logits for the exponent floor
    abs_logits = torch.abs(logits)
    
    # Step 3: Implement the stable per-element component equation
    # max(x, 0) - x * y + log(1 + exp(-|x|))
    per_element_loss = max_val - (logits * targets) + torch.log(1.0 + torch.exp(-abs_logits))
    
    # Step 4: Average over all elements in the batch
    mean_loss = torch.mean(per_element_loss)
    
    # Step 5: Extract raw float and round neatly to 4 decimal places
    return round(float(mean_loss.item()), 4)
