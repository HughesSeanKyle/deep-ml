import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    # Convert list to a numpy array for vectorised operations
    probs = np.array(probabilities)
    
    # Calculate the natural logarithm of each probability
    log_probs = np.log(probs)
    
    # Compute the average log-likelihood
    avg_log_prob = np.mean(log_probs)
    
    # Perplexity is exp(-avg_log_prob)
    perplexity = np.exp(-avg_log_prob)
    
    return float(perplexity)