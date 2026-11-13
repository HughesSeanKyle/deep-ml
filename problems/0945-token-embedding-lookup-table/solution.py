import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list.
    """
    # Initialize the random number generator with the given seed
    rng = np.random.default_rng(seed)
    
    # Create the embedding weight matrix of shape (vocab_size, embed_dim)
    embedding_table = rng.standard_normal((vocab_size, embed_dim))
    
    # Handle empty token_ids gracefully while preserving expected 2D shape
    if not token_ids:
        return np.empty((0, embed_dim)).tolist()
        
    # Use NumPy integer array indexing to look up the rows for token_ids
    selected_embeddings = embedding_table[token_ids]
    
    return selected_embeddings.tolist()