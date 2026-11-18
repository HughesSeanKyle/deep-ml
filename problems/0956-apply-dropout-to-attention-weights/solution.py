import numpy as np

def attention_dropout(attn_weights, values, dropout_rate: float, mask) -> list[list[float]]:
    """Apply dropout mask to attention weights and compute context vectors."""
    attn_weights = np.asarray(attn_weights, dtype=float)
    values = np.asarray(values, dtype=float)
    
    # If dropout_rate is 0, skip masking and scaling entirely
    if dropout_rate == 0:
        dropped_weights = attn_weights
    else:
        mask = np.asarray(mask, dtype=float)
        # Zero out dropped entries and scale surviving entries by 1 / (1 - dropout_rate)
        dropped_weights = (attn_weights * mask) / (1.0 - dropout_rate)
        
    # Multiply the post-dropout attention weights with the value matrix: shape (T, d_v)
    context = dropped_weights @ values
    
    return context.tolist()
