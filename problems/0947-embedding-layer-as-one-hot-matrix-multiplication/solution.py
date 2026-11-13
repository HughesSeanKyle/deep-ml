import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    W = np.asarray(W, dtype=float)
    num_tokens = len(token_ids)
    vocab_size = W.shape[0]
    
    # Initialize the one-hot matrix H of shape (num_tokens, vocab_size) with zeros
    H = np.zeros((num_tokens, vocab_size), dtype=float)
    
    # Set column token_ids[i] to 1.0 for each row i
    if num_tokens > 0:
        H[np.arange(num_tokens), token_ids] = 1.0
        
    # Perform matrix multiplication H @ W to extract the corresponding embedding rows
    return H @ W