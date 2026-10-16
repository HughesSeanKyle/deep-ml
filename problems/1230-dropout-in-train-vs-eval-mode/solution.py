import torch
import torch.nn as nn


def dropout_demo():
    """Demonstrate Dropout behavior in eval vs train mode.

    Returns:
        tuple: (eval_output, train_nonzero_count)
            eval_output: result of Dropout(ones) in eval mode (identity)
            train_nonzero_count: int count of nonzero elements after Dropout in train mode
    """
    # 1. Set seed for absolute RNG determinism
    torch.manual_seed(0)
    
    # 2. Build input tensor of 10 ones and initialize the Dropout layer
    x = torch.ones(10)
    drop = nn.Dropout(p=0.5)
    
    # 3. Transition module to evaluation mode
    drop.eval()
    eval_output = drop(x)  # Acts as a no-op (identity mapping)
    
    # 4. Transition module back to training mode
    drop.train()
    train_output = drop(x)  # Randomly zeros out elements and scales survivors to 2.0
    
    # 5. Count the remaining nonzero entries using a boolean mask
    # .item() converts a 0-dimensional tensor into a standard Python int
    train_nonzero_count = int(torch.sum(train_output != 0).item())
    
    return eval_output, train_nonzero_count
