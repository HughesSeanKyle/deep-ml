import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Creates a tensor of shape (seq_len, d_model) holding sinusoidal positional encodings.
    Assumes d_model is even.
    """
    # 1. Initialize an empty tensor to hold the final embedding table
    pe = torch.zeros(seq_len, d_model)
    
    # 2. Build a column vector of positions: shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # 3. Compute division terms in log space for stability: shape (d_model // 2,)
    # We step by 2 because each index i covers both a sine and a cosine component
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
    )
    
    # 4. Use an outer product to calculate angles across all features
    # Shape: (seq_len, d_model // 2)
    angles = position * div_term
    
    # 5. Populate even columns (0, 2, 4...) with sine waves
    pe[:, 0::2] = torch.sin(angles)
    
    # 6. Populate odd columns (1, 3, 5...) with cosine waves
    pe[:, 1::2] = torch.cos(angles)
    
    return pe
