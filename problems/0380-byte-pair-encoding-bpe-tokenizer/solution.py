from collections import defaultdict

def byte_pair_encoding(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    if not corpus or num_merges <= 0:
        return []

    # Convert space-separated string keys into lists of tokens for easier manipulation
    # Structure: [(token_list, frequency), ...]
    vocab_words = [(word.split(), freq) for word, freq in corpus.items()]
    merges = []

    for _ in range(num_merges):
        pair_counts = defaultdict(int)

        # 1. Count all adjacent pairs weighted by word frequency
        for tokens, freq in vocab_words:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq

        # Stop early if no adjacent pairs exist in any word
        if not pair_counts:
            break

        # 2. Find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)

        # 3. Merge the winning pair everywhere across all words
        target_a, target_b = best_pair
        merged_token = target_a + target_b
        new_vocab_words = []

        for tokens, freq in vocab_words:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == target_a and tokens[i + 1] == target_b:
                    new_tokens.append(merged_token)
                    i += 2  # Non-overlapping skip
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab_words.append((new_tokens, freq))

        vocab_words = new_vocab_words

    return merges