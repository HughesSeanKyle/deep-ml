import torch

def train_step(model, x, y, optimizer, loss_fn):
    """
    Run one training step and return the pre-update loss as a float.

    Args:
        model: torch.nn.Module to train.
        x: Input batch tensor.
        y: Target batch tensor.
        optimizer: torch.optim optimizer bound to model parameters.
        loss_fn: Callable (pred, y) -> scalar loss tensor.

    Returns:
        float: Loss value computed before optimizer.step().
    """
    # Step 1: Clear out stale parameter gradients from previous loops
    optimizer.zero_grad()
    
    # Step 2: Compute the forward pass through the model pipeline
    pred = model(x)
    
    # Step 3: Evaluate the loss tensor using the specified loss objective
    loss = loss_fn(pred, y)
    
    # Step 4: Extract the loss scalar value as a Python float BEFORE parameters mutate
    loss_val = float(loss.item())
    
    # Step 5: Execute backpropagation to accumulate gradients
    loss.backward()
    
    # Step 6: Step the optimizer forward to adjust model weights
    optimizer.step()
    
    # Return the clean float loss snapshot
    return loss_val
