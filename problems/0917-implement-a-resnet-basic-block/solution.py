import os
# Silences low-level C++ engine warnings from PyTorch backends
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # 1. First structural block: Downsamples spatial layout if stride > 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 2. Second structural block: Maintains standard feature dimensions (stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 3. Dynamic Shortcut Selection Block
        # If sizes and channels align perfectly, use an identity passthrough.
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            # If shape components differ, use a 1x1 projection conv to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Main block 1 processing
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Step 2: Main block 2 processing (stops right before its activation)
        out = self.bn2(self.conv2(out))
        
        # Step 3: Compute shortcut connection state and perform the highway addition
        out = out + self.shortcut(x)
        
        # Step 4: Apply final ReLU activation to the merged representation
        return self.relu(out)
