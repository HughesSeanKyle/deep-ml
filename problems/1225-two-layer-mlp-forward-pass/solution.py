import torch
import torch.nn as nn


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    """
    Build a 2-layer MLP, set fixed weights, return scalar output.
    """
    # Step 1: Construct the sequential structural model container
    mlp = nn.Sequential(
        nn.Linear(2, 2),  # Index 0
        nn.ReLU(),        # Index 1
        nn.Linear(2, 1)   # Index 2
    )
    
    # Step 2: Use in-place parameter copies under no_grad to fix weights
    with torch.no_grad():
        # First Linear layer assignment (mlp[0])
        mlp[0].weight.copy_(w1)
        mlp[0].bias.copy_(b1)
        
        # Second Linear layer assignment (mlp[2])
        mlp[2].weight.copy_(w2)
        mlp[2].bias.copy_(b2)
        
    # Step 3: Run the input forward through the sequential pipeline
    output_tensor = mlp(x)
    
    # Step 4: Extract and return the value as a native Python float
    return float(output_tensor.item())
