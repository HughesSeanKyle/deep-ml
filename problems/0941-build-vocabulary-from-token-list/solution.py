def build_vocab(tokens: list[str]) -> dict[str, int]:
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    if not tokens:
        return {}
        
    # Get unique tokens and sort them lexicographically
    unique_sorted_tokens = sorted(set(tokens))
    
    # Map each sorted token to its index/ID
    vocab = {token: idx for idx, token in enumerate(unique_sorted_tokens)}
    
    return vocab
