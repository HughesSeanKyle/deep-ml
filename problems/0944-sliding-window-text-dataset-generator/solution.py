def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    dataset = []
    n = len(token_ids)
    
    # Iterate i starting at 0, advancing by stride, while the target chunk fits completely
    for i in range(0, n - max_length, stride):
        input_chunk = list(token_ids[i : i + max_length])
        target_chunk = list(token_ids[i + 1 : i + max_length + 1])
        dataset.append((input_chunk, target_chunk))
        
    return dataset