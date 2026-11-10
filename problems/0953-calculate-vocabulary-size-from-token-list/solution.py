def vocab_size(tokens: list[str], special_tokens: list[str] = None) -> int:
    """Return the vocabulary size from a list of preprocessed tokens.

    Args:
        tokens: list[str] of preprocessed tokens
        special_tokens: optional list[str] of special tokens to include
    Returns:
        int: number of unique tokens in the combined vocabulary
    """
    # Create a set of unique tokens from the main corpus
    unique_tokens = set(tokens)
    
    # If special tokens are provided, union them into the set
    if special_tokens:
        unique_tokens.update(special_tokens)
        
    return len(unique_tokens)