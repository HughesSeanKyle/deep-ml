def pad_and_truncate(sequences: list[list[int]], pad_token_id: int, max_length: int | None = None) -> list[list[int]]:
    """
    Pad and/or truncate tokenized sequences to a uniform length.

    Args:
        sequences: list of lists of int token ids
        pad_token_id: int, token id used for padding
        max_length: int or None, target length. If None, use the longest sequence length.

    Returns:
        list of lists of int token ids, each of length max_length.
    """
    if not sequences:
        return []
        
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
        
    if max_length == 0:
        return [[] for _ in sequences]
        
    padded_sequences = []
    for seq in sequences:
        # Truncate if longer than max_length
        truncated = list(seq[:max_length])
        # Calculate how many pad tokens are needed and append them
        pad_count = max_length - len(truncated)
        padded_sequences.append(truncated + [pad_token_id] * pad_count)
        
    return padded_sequences
