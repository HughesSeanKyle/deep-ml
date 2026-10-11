import torch
import torch.nn as nn


def single_neuron_forward(x):
    """
    Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    # Step 1: Create a linear layer with 3 inputs and 1 output
    neuron = nn.Linear(3, 1)
    
    # Step 2: Fix parameters under torch.no_grad context
    with torch.no_grad():
        # Define our target fixed weights and bias
        target_weight = torch.tensor([[0.5, -0.2, 0.3]])
        target_bias = torch.tensor([0.1])
        
        # In-place copy into the module parameters to preserve object graph
        neuron.weight.copy_(target_weight)
        neuron.bias.copy_(target_bias)
        
    # Step 3: Run the forward pass on input x
    output_tensor = neuron(x)
    
    # Step 4: Convert the result from a single-element tensor to a native float
    return float(output_tensor.item())
