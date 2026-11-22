import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
    """
    Compute temperature-scaled softmax probabilities from logits.
    
    Args:
        logits: 1D numpy array of raw model output scores
        temperature: float controlling distribution sharpness
    
    Returns:
        List of probabilities after temperature scaling
    """
    logits = np.asarray(logits, dtype=float)
    
    # Greedy decoding: when temperature <= 0, put all probability on the first max logit
    if temperature <= 0:
        one_hot = np.zeros_like(logits, dtype=float)
        one_hot[np.argmax(logits)] = 1.0
        return one_hot.tolist()
        
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Numerically stable softmax
    shifted_logits = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits)
    
    return probs.tolist()