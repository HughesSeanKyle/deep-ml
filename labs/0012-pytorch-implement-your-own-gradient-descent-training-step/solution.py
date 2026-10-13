import torch
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, x_batch, y_batch, lr):
    """
    Perform ONE step of gradient descent training manually without torch.optim.
    """
    # Step 1: Zero all gradients to prevent accumulation across batches
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
            
    # Step 2: Forward pass through the network model graph
    logits = model(x_batch)
    
    # Step 3: Compute classification loss using cross entropy
    loss = F.cross_entropy(logits, y_batch)
    
    # Step 4: Extract the loss scalar as a float BEFORE parameter mutation
    # Caching it handles memory cleanly and avoids retaining graph histories
    loss_val = float(loss.item())
    
    # Step 5: Backward pass to compute the parameter gradients
    loss.backward()
    
    # Step 6: Update all parameters manually under a no_grad context
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                # Core gradient descent formula: new = old - lr * gradient
                param -= lr * param.grad
                
    return loss_val
