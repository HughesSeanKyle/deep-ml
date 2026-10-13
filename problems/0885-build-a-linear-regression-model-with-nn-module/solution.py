import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        # Step 1: Call the parent constructor (nn.Module) to enable internal bookkeeping
        super().__init__()
        
        # Step 2: Register a single nn.Linear layer as an instance attribute
        # PyTorch automatically intercepts this assignment and registers its parameters
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 3: Pass the incoming data tensor through the registered linear layer
        return self.linear(x)
