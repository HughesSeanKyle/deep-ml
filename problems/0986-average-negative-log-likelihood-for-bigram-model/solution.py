import numpy as np

def bigram_avg_nll(P, words, stoi):
    """Average negative log-likelihood of words under a bigram model."""
    if not words:
        return 0.0
        
    P = np.asarray(P, dtype=float)
    total_log_prob = 0.0
    total_bigrams = 0
    
    for w in words:
        padded = '.' + w + '.'
        for c1, c2 in zip(padded, padded[1:]):
            ix1 = stoi[c1]
            ix2 = stoi[c2]
            prob = P[ix1, ix2]
            total_log_prob += np.log(prob)
            total_bigrams += 1
            
    if total_bigrams == 0:
        return 0.0
        
    avg_nll = -total_log_prob / total_bigrams
    return round(float(avg_nll), 4)
