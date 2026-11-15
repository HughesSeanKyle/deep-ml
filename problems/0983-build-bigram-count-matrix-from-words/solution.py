import numpy as np

def bigram_counts(words: list[str]) -> list[list[int]]:
    """
    Build a V x V bigram count matrix from a list of words, using '.' as the
    start and end boundary token at index 0.
    """
    # 1. Collect all unique characters excluding '.', sort them, and prepend '.'
    unique_chars = sorted(set("".join(words)) - {'.'})
    vocab = ['.'] + unique_chars
    
    # Map each token to its integer index
    stoi = {ch: idx for idx, ch in enumerate(vocab)}
    
    # 2. Initialize a V x V integer matrix of zeros
    V = len(vocab)
    N = np.zeros((V, V), dtype=int)
    
    # 3. Augment each word with boundary tokens and count consecutive pairs
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
            
    return N.tolist()