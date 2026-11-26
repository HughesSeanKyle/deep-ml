import numpy as np

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
    """
    Apply a combined sampling pipeline to raw logits.
    
    Args:
        logits: Raw unnormalized scores for each token
        temperature: Scaling factor for logit sharpness (0 = greedy)
        top_k: Number of top tokens to keep (0 = disabled)
        top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
        seed: Random seed for reproducible sampling
    
    Returns:
        Dict with 'probabilities' (list of floats rounded to 4 decimals)
        and 'sampled_token' (int index of chosen token)
    """
    logits_arr = np.asarray(logits, dtype=np.float64)
    n = logits_arr.size

    # 1. Greedy decoding: return one-hot at first max index (no RNG call)
    if temperature <= 0:
        best_idx = int(np.argmax(logits_arr))
        probs = np.zeros(n, dtype=np.float64)
        probs[best_idx] = 1.0
        return {
            "probabilities": [round(float(p), 4) for p in probs],
            "sampled_token": best_idx,
        }

    # 2. Temperature scaling
    z = logits_arr / temperature

    # 3. Top-k filtering (keep top_k largest z_i; mask others to -inf)
    if 0 < top_k < n:
        # Use lexsort so ties deterministically prefer smaller token indices
        sorted_indices = np.lexsort((np.arange(n), -z))
        keep_indices = sorted_indices[:top_k]
        masked_z = np.full(n, -np.inf, dtype=np.float64)
        masked_z[keep_indices] = z[keep_indices]
        z = masked_z

    # 4. Numerically stable softmax over masked z
    max_z = np.max(z)
    exp_z = np.exp(z - max_z)
    probs = exp_z / np.sum(exp_z)

    # 5. Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.lexsort((np.arange(n), -probs))
        sorted_probs = probs[sorted_indices]
        
        # Cumulative sum strictly BEFORE each token: [0, p_0, p_0 + p_1, ...]
        cum_probs_before = np.cumsum(sorted_probs) - sorted_probs
        keep_mask = cum_probs_before < top_p
        
        # Ensure at least the top token is kept (handles edge case top_p <= 0)
        keep_mask[0] = True
        
        filtered_probs = np.zeros(n, dtype=np.float64)
        kept_indices = sorted_indices[keep_mask]
        filtered_probs[kept_indices] = probs[kept_indices]
        probs = filtered_probs / np.sum(filtered_probs)

    # 6. Sample from unrounded final distribution (single RNG call)
    rng = np.random.default_rng(seed)
    sampled_token = int(rng.choice(n, p=probs))

    # 7. Round probabilities after sampling
    return {
        "probabilities": [round(float(p), 4) for p in probs],
        "sampled_token": sampled_token,
    }