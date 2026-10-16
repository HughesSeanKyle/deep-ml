import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """
    Apply inverted dropout to an arbitrary shaped input tensor x.
    
    Args:
        x: Input tensor of any shape
        p: Probability of an element being zeroed (0 <= p < 1)
        training: If True, apply dropout; if False, return x unchanged
        
    Returns:
        Regularised tensor of the same shape as x
    """
    # 1. Immediate exit check for evaluation mode or zero probability
    if not training or p == 0.0:
        return x
        
    # 2. Generate a random uniform tensor of the same shape as x
    # Elements are uniformly distributed between [0, 1)
    random_tensor = torch.rand_like(x)
    
    # 3. Create a binary mask where True (1.0) means KEEP and False (0.0) means DROP
    # Since we want elements to drop out with probability p, they must survive with (1 - p).
    # Thus, checking if the random variable is >= p yields exactly a 1 - p success rate.
    mask = (random_tensor >= p).float()
    
    # 4. Compute inverted dropout: mask the input and scale up the survivors
    scale = 1.0 / (1.0 - p)
    y = x * mask * scale
    
    return y
