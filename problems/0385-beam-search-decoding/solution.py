import numpy as np

def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, V = log_probs.shape
    
    if T == 0:
        return []

    # Step 0: sort by (-score, token_index) so ties deterministically prefer smaller token indices
    k_init = min(beam_width, V)
    init_scores = log_probs[0]
    top_init_indices = np.lexsort((np.arange(V), -np.round(init_scores, 12)))[:k_init]
    
    beams = [[int(idx)] for idx in top_init_indices]
    scores = init_scores[top_init_indices]

    # Steps 1 .. T-1
    for t in range(1, T):
        # (num_beams, V) -> flattened in row-major order (beam 0 tokens, beam 1 tokens, ...)
        candidate_scores = scores[:, None] + log_probs[t][None, :]
        flat_scores = candidate_scores.ravel()
        
        k_step = min(beam_width, flat_scores.size)
        # Primary key: -score (descending), Secondary key: flat index (ascending)
        top_flat_indices = np.lexsort((np.arange(flat_scores.size), -np.round(flat_scores, 12)))[:k_step]
        
        beam_indices = top_flat_indices // V
        token_indices = top_flat_indices % V
        
        beams = [beams[b_idx] + [int(t_idx)] for b_idx, t_idx in zip(beam_indices, token_indices)]
        scores = flat_scores[top_flat_indices]

    num_return = min(top_k, len(beams))
    return [
        (beams[i], round(float(scores[i]), 4))
        for i in range(num_return)
    ]