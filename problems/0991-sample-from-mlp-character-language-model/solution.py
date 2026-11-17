import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size: int, itos: dict[int, str], seed: int, max_tokens: int) -> str:
    """
    Sample a string autoregressively from a trained character-level MLP language model.
    """
    C = np.asarray(C, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    
    vocab_size = C.shape[0]
    rng = np.random.default_rng(seed)
    
    # Initialize context window with block_size start/end tokens (index 0)
    context = [0] * block_size
    generated_chars = []
    
    for _ in range(max_tokens):
        # 1. Embed context characters and flatten into a single 1D vector
        e = C[context].reshape(-1)
        
        # 2. Forward pass through hidden tanh layer and output linear layer
        h = np.tanh(e @ W1 + b1)
        logits = h @ W2 + b2
        
        # 3. Numerically stable softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        
        # 4. Sample next token index from the probability distribution
        ix = int(rng.choice(vocab_size, p=probs))
        
        # Stop generation if the end token (0) is drawn
        if ix == 0:
            break
            
        generated_chars.append(itos[ix])
        
        # Slide the context window forward: drop oldest token, append newly sampled token
        context = context[1:] + [ix]
        
    return "".join(generated_chars)