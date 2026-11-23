import numpy as np

def greedy_stream(model, prompt_ids, max_new_tokens: int, eos_id: int) -> list:
    """
    Greedy streaming decoder backed by a KV cache.

    Args:
        model: object exposing prefill(prompt_ids) -> np.ndarray of logits
               and decode_step(token_id) -> np.ndarray of logits.
        prompt_ids: iterable of int token ids.
        max_new_tokens: maximum number of tokens to generate.
        eos_id: token id that terminates generation.

    Returns:
        List of generated token ids (excluding the prompt and EOS).
    """
    # 1. Always call prefill exactly once with the full prompt to prime the KV cache
    logits = model.prefill(list(prompt_ids))
    
    generated = []
    
    # 2. Decode phase: runs 0 times if max_new_tokens <= 0
    for _ in range(max_new_tokens):
        next_token = int(np.argmax(logits))
        
        # Stop immediately if EOS token is generated (without appending or stepping)
        if next_token == eos_id:
            break
            
        generated.append(next_token)
        
        # Advance the KV cache by one single token to get the next logits
        logits = model.decode_step(next_token)
        
    return generated
