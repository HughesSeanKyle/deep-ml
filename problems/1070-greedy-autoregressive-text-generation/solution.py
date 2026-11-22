import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    """
    Generate tokens autoregressively using greedy decoding (argmax at each step).
    """
    generated = list(idx)
    
    for _ in range(max_new_tokens):
        # 1. Crop the sequence to the last context_size tokens and add batch dim: (1, T)
        idx_cond = np.array([generated[-context_size:]], dtype=np.int64)
        
        # 2. Forward pass through the model to get logits: shape (1, T, vocab_size)
        logits = model(idx_cond)
        
        # 3. Extract logits at the final time step: shape (1, vocab_size)
        last_step_logits = logits[:, -1, :]
        
        # 4. Pick the highest-scoring token ID and extract the scalar from shape (1,)
        next_token = int(np.argmax(last_step_logits, axis=-1)[0])
        
        # 5. Append to the running sequence
        generated.append(next_token)
        
    return generated