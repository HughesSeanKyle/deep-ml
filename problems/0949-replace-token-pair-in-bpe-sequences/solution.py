def replace_pair(sequences: list[list[int]], pair: tuple[int, int], new_id: int) -> list[list[int]]:
    """
    Scans each sequence left to right and replaces every non-overlapping 
    occurrence of the adjacent pair (a, b) with new_id.
    """
    target_a, target_b = pair
    updated_sequences = []
    
    for seq in sequences:
        # Sequences shorter than 2 can't contain an adjacent pair
        if len(seq) < 2:
            updated_sequences.append(seq.copy())
            continue
            
        new_seq = []
        i = 0
        
        while i < len(seq):
            # Check if we have room for a pair and it matches our target pair
            if i < len(seq) - 1 and seq[i] == target_a and seq[i+1] == target_b:
                new_seq.append(new_id)
                i += 2  # Skip past both merged tokens to avoid overlapping reuse
            else:
                new_seq.append(seq[i])
                i += 1  # Advance normally
                
        updated_sequences.append(new_seq)
        
    return updated_sequences