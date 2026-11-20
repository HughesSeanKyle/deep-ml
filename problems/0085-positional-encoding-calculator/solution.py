import numpy as np

def pos_encoding(position: int, d_model: int):
    # Guard against invalid sequence lengths or model dimensions
    if position <= 0 or d_model <= 0:
        return -1

    # Column vector of positions: shape (position, 1)
    pos = np.arange(position, dtype=np.float64)[:, np.newaxis]
    
    # Row vector of dimension indices: shape (1, d_model)
    i = np.arange(d_model, dtype=np.float64)[np.newaxis, :]
    
    # Compute the angular frequencies: 1 / 10000^(2 * (i // 2) / d_model)
    # Using i // 2 ensures even index 2k and odd index 2k+1 share the exact same frequency
    angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / d_model)
    
    # Broadcast multiply to get angle matrix of shape (position, d_model)
    angles = pos * angle_rates
    
    # Apply sin to even columns (0, 2, 4, ...) and cos to odd columns (1, 3, 5, ...)
    pe = np.zeros_like(angles)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    pos_encoding = np.float16(pe)
    return pos_encoding