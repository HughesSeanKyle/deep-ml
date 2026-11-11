def bpe_encode(text: str, token_to_id: dict[str, int], merges: dict[tuple[int, int], int]) -> list[int]:
    """
    Encode text into subword token IDs using a trained BPE merge table.
    """
    if not text:
        return []
        
    # Step 1: Convert raw text into initial list of character-level token IDs
    ids = [token_to_id[char] for char in text]
    
    # Step 2: Repeatedly apply the highest-priority valid merge until none remain
    while len(ids) >= 2:
        best_pair = None
        min_rank = float('inf')
        
        # Scan current sequence to find the valid merge with the lowest new_id (highest priority)
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            if pair in merges:
                rank = merges[pair]
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
                    
        # If no adjacent pairs in the current sequence exist in the merge table, we are done
        if best_pair is None:
            break
            
        # Step 3: Replace all non-overlapping left-to-right occurrences of best_pair
        new_ids = []
        i = 0
        target_a, target_b = best_pair
        
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == target_a and ids[i+1] == target_b:
                new_ids.append(min_rank)
                i += 2  # Skip both merged tokens to prevent overlapping reuse
            else:
                new_ids.append(ids[i])
                i += 1
                
        ids = new_ids
        
    return ids