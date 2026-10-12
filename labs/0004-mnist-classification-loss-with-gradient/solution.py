import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    Computes Cross-Entropy loss and its gradient with respect to 
    the input softmax probabilities.
    
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: "mean", "sum", or "none"
    """
    # Step 1: Validate shapes and values early
    assert preds.ndim == 2, "Predictions must be a 2D matrix"
    N, C = preds.shape
    assert target.shape == (N,), "Target must match batch dimension N"
    
    # Step 2: Safe clipping to prevent log(0) and division-by-zero crashes
    eps = 1e-12
    p_stable = np.clip(preds, eps, 1.0)
    
    # Step 3: Compute the per-sample loss vector using advanced indexing
    # We grab the probability corresponding to the true target index for each row
    correct_class_probs = p_stable[np.arange(N), target]
    per_sample_loss = -np.log(correct_class_probs)
    
    # Step 4: Calculate the dL/dp unscaled gradient matrix
    # Build a mask containing 0 everywhere except for 1 at the target labels
    T = np.zeros_like(preds)
    T[np.arange(N), target] = 1.0
    
    # Core derivative formula: dL/dp = -T / p
    grad = -T / p_stable
    
    # Step 5: Handle reductions and compute the corresponding gradient scales
    if reduction == "mean":
        loss = float(per_sample_loss.mean())
        scale = 1.0 / N
    elif reduction == "sum":
        loss = float(per_sample_loss.sum())
        scale = 1.0
    else:  # "none"
        loss = per_sample_loss
        scale = 1.0
        
    # Step 6: Apply the reduction scale back onto the gradient matrix
    grad *= scale
    
    return loss, grad