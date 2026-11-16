import numpy as np

def mlp_forward_loss(X, Y, C, W1, b1, W2, b2) -> float:
    """
    Forward pass of an MLP character-level language model.
    Returns mean cross-entropy loss as a float.
    """
    X = np.asarray(X, dtype=int)
    Y = np.asarray(Y, dtype=int)
    C = np.asarray(C, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    
    N, block_size = X.shape
    emb_dim = C.shape[1]
    
    # 1. Look up embeddings: shape (N, block_size, emb_dim)
    emb = C[X]
    
    # 2. Concatenate (flatten) context embeddings per example: shape (N, block_size * emb_dim)
    flat = emb.reshape(N, block_size * emb_dim)
    
    # 3. Hidden layer with tanh activation: shape (N, hidden)
    h = np.tanh(flat @ W1 + b1)
    
    # 4. Output logits: shape (N, vocab_size)
    logits = h @ W2 + b2
    
    # 5. Numerically stable log-softmax and negative log-likelihood loss
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp
    
    # Extract log probabilities for the true target indices Y and compute mean NLL
    target_log_probs = log_probs[np.arange(N), Y]
    loss = -np.mean(target_log_probs)
    
    return float(loss)
