# --- cell 1 ---
from collections import Counter


def train_tokenizer(corpus, vocab_size):
    """
    Build a Byte Pair Encoding (BPE) tokenizer from the training corpus.
    """
    # 1. Initialize base vocabulary with all unique characters in sorted order
    unique_chars = sorted(set("".join(corpus)))
    
    # Map token string -> integer ID, and integer ID -> token string
    token_to_id = {ch: idx for idx, ch in enumerate(unique_chars)}
    id_to_token = {idx: ch for idx, ch in enumerate(unique_chars)}
    
    # Represent each unique word in the corpus with its frequency
    # Using word frequencies makes training on 25,000 names ~10x faster
    word_freqs = Counter(corpus)
    splits = {word: list(word) for word in word_freqs}
    
    # Store ordered merge rules: (token_a, token_b) -> merged_token
    merges = []
    
    # 2. Iteratively merge the most frequent adjacent pair up to vocab_size
    while len(token_to_id) < vocab_size:
        pair_counts = Counter()
        
        for word, freq in word_freqs.items():
            tokens = splits[word]
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += freq
                
        if not pair_counts:
            break
            
        # Find the most frequent pair (tie-break lexicographically for determinism)
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merged_token = best_pair[0] + best_pair[1]
        
        # Register the new subword token in our vocabulary
        new_id = len(token_to_id)
        token_to_id[merged_token] = new_id
        id_to_token[new_id] = merged_token
        merges.append(best_pair)
        
        # Apply the merge across all word splits
        for word in splits:
            tokens = splits[word]
            if len(tokens) < 2:
                continue
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            splits[word] = new_tokens

    # Create a rank lookup dictionary so encode() can find the highest-priority merge fast
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    # 3. Define lossless encode and decode closures
    def encode(text: str) -> list[int]:
        if not text:
            return []
            
        tokens = list(text)
        
        while len(tokens) >= 2:
            best_pair = None
            best_rank = float("inf")
            
            # Find the adjacent pair learned earliest during training
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    
            if best_pair is None:
                break
                
            # Replace all non-overlapping occurrences of best_pair left-to-right
            merged_token = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return [token_to_id[t] for t in tokens]

    def decode(token_ids: list[int]) -> str:
        return "".join(id_to_token[idx] for idx in token_ids)

    return encode, decode