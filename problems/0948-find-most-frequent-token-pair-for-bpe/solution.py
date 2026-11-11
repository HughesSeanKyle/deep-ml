from collections import defaultdict

def most_frequent_pair(sequences: list[list[int]]) -> tuple[int, int] | None:
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    pair_counts = defaultdict(int)
    first_appearance = {}
    order_index = 0  # Global chronological tracker for first appearance

    # Step 1: Scan and count all adjacent pairs
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
            
            # Record the very first time we encounter this specific pair
            if pair not in first_appearance:
                first_appearance[pair] = order_index
            order_index += 1

    if not pair_counts:
        return None

    # Step 2: Find the best pair matching both rules
    best_pair = None
    max_count = -1

    for pair, count in pair_counts.items():
        if count > max_count:
            max_count = count
            best_pair = pair
        elif count == max_count:
            # Tiebreaker rule: pick the pair that showed up earliest
            if first_appearance[pair] < first_appearance[best_pair]:
                best_pair = pair

    return best_pair