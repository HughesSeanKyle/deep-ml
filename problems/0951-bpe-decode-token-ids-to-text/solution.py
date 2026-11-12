def bpe_decode(ids: list[int], vocab: dict[int, str]) -> str:
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    if not ids:
        return ""
        
    decoded_pieces = []
    
    for token_id in ids:
        token = vocab[token_id]
        
        # Replace only a leading 'G' with a single space
        if token.startswith('G'):
            token = ' ' + token[1:]
            
        decoded_pieces.append(token)
        
    return "".join(decoded_pieces)