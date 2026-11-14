def make_input_target(token_ids: list[int], context_size: int) -> list[list[int]]:
    """Build next-token prediction (input, target) pair.

    Args:
        token_ids: list[int] of token IDs (length >= context_size + 1)
        context_size: int, number of tokens in the input window
    Returns:
        list: [x, y] where x and y are lists of length context_size,
              with y being x shifted right by one position.
    """
    if len(token_ids) < context_size + 1:
        raise ValueError(
            f"token_ids must have at least {context_size + 1} elements, got {len(token_ids)}."
        )
        
    if context_size == 0:
        return [[], []]
        
    x = token_ids[:context_size]
    y = token_ids[1 : context_size + 1]
    
    return [x, y]