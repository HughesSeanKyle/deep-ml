import numpy as np

def learned_positional_encoding(token_embeddings: np.ndarray, position_embedding_table: np.ndarray, start_pos: int = 0) -> np.ndarray:
    """
    Apply learned positional embeddings to token embeddings.
    
    Args:
        token_embeddings: (batch_size, seq_len, d_model) array of token embeddings
        position_embedding_table: (max_seq_len, d_model) learned positional embedding lookup table
        start_pos: Starting position index (default 0)
    
    Returns:
        Array of shape (batch_size, seq_len, d_model) with positional information applied
    """
    token_embeddings = np.asarray(token_embeddings, dtype=float)
    position_embedding_table = np.asarray(position_embedding_table, dtype=float)
    
    seq_len = token_embeddings.shape[1]
    
    # Validate that start_pos + seq_len fits within max_seq_len
    if start_pos < 0 or start_pos + seq_len > position_embedding_table.shape[0]:
        raise ValueError(
            f"Position slice [{start_pos}:{start_pos + seq_len}] exceeds table length {position_embedding_table.shape[0]}."
        )
        
    # Slice the positional embeddings for positions [start_pos, start_pos + seq_len)
    pos_emb = position_embedding_table[start_pos : start_pos + seq_len]
    
    # Add element-wise; NumPy broadcasts (seq_len, d_model) across batch_size
    return token_embeddings + pos_emb