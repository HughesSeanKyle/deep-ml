import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """A standard two-layer residual block with BatchNorm and GELU."""
    def __init__(self, size: int):
        super().__init__()
        # 2 Linear layers inside this block = 2 layers added to total depth
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.GELU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F(x) + x shortcut mechanism
        return self.activation(self.block(x) + x)

class DeepNetwork(nn.Module):
    """A highly performant 32-layer ResNet architecture built for CPU efficiency."""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        
        # 1. Input projection layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )
        
        # 2. Hidden core: 15 Residual Blocks * 2 linear layers each = 30 hidden layers
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(15)
        ])
        
        # 3. Output classification layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        # Apply Kaiming Normal initialization to facilitate clean startup dynamics
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input from shape (B, 28, 28) or (B, 784) safely into (B, 784)
        x = x.view(x.size(0), -1)
        
        # Input layer mapping
        x = self.input_layer(x)
        
        # Loop sequentially through the 15 deep residual blocks
        for block in self.layers:
            x = block(x)
            
        # Final classification scores
        return self.output_layer(x)
