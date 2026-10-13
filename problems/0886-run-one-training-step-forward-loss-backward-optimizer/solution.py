import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    """
    Performs exactly one step of stochastic gradient descent (SGD) 
    using mean-squared-error (MSE) loss on the provided model.
    """
    # 1. Instantiate the optimizer fresh over the model parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 2. Clear out any stale gradients remaining from prior computations
    optimizer.zero_grad()
    
    # 3. Execute the forward pass through the model graph
    preds = model(x)
    
    # 4. Compute the scalar MSE loss
    loss = F.mse_loss(preds, y)
    
    # 5. Extract the loss value as a native Python float BEFORE the parameter update
    loss_val = float(loss.item())
    
    # 6. Run backpropagation to populate gradients (.grad attributes)
    loss.backward()
    
    # 7. Step the optimizer forward to adjust model weights
    optimizer.step()
    
    return loss_val
