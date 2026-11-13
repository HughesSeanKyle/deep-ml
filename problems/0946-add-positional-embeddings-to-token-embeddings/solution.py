import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""
    # Extract the sequence length from the second dimension of token_embeddings
    seq_len = token_embeddings.shape[1]
    
    # Slice the first seq_len rows: shape becomes (seq_len, embed_dim)
    pos_embeddings = pos_embedding_matrix[:seq_len]
    
    # Add element-wise; NumPy automatically broadcasts (seq_len, embed_dim)
    # across (batch_size, seq_len, embed_dim)
    return token_embeddings + pos_embeddings
