import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    logits_arr = np.asarray(logits, dtype=float)
    
    # 1. Numerically stable softmax to get initial probability distribution
    shifted_logits = logits_arr - np.max(logits_arr)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits)
    
    # 2. Sort token indices by descending probability, breaking ties by smaller index
    # Using lexsort on (indices, -probs) sorts primarily by -probs ascending, then indices ascending
    indices = np.arange(len(probs))
    sorted_indices = np.lexsort((indices, -probs))
    sorted_probs = probs[sorted_indices]
    
    # 3. Compute cumulative probability mass and find the cutoff index where cumsum >= p
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_idx = int(np.searchsorted(cumulative_probs, p, side='left'))
    # Clamp cutoff_idx in case floating-point rounding puts total sum slightly below 1.0 (e.g. 0.9999999)
    cutoff_idx = min(cutoff_idx, len(probs) - 1)
    
    # 4. Keep tokens up to and including cutoff_idx, zero out the rest
    nucleus_indices = sorted_indices[: cutoff_idx + 1]
    filtered_probs = np.zeros_like(probs)
    filtered_probs[nucleus_indices] = probs[nucleus_indices]
    
    # 5. Renormalize surviving probabilities so they sum to 1.0
    filtered_probs /= np.sum(filtered_probs)
    
    return np.round(filtered_probs, 4).tolist()