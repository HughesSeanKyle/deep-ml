import torch
import torch.nn as nn

def count_params() -> int:
    """
    Builds a fixed Sequential model (Linear(4,8) -> ReLU -> Linear(8,2)) 
    and returns the total count of trainable parameters.
    """
    # Step 1: Instantiate the target multi-layer sequential model pipeline
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    
    # Step 2: Use a generator expression to sum up p.numel() 
    # for all parameters where requires_grad is True
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return int(total_trainable_params)
