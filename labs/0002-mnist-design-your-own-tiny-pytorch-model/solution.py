# --- cell 1 ---
import torch
import torch.nn as nn

def build_model() -> nn.Module:
    """
    An optimized tiny network targeting exactly ~2,010 parameters.
    Maximizes training capacity under a 2,048 limit for small sample learning.
    """
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Stage 1: Standard Convolution (extracts foundational features)
            self.conv1 = nn.Conv2d(1, 28, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(28)
            self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14
            
            # Stage 2: Depthwise-Separable Convolution (Efficient Channel Expansion)
            self.depthwise = nn.Conv2d(28, 28, kernel_size=3, padding=1, groups=28, bias=False)
            self.pointwise = nn.Conv2d(28, 36, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(36)
            self.pool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7
            
            # Activations and Pooling
            self.act = nn.ReLU()
            self.gap = nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1
            
            # Classification Head
            self.fc = nn.Linear(36, 10)

        def forward(self, x):
            # First feature block
            x = self.pool1(self.bn1(self.act(self.conv1(x))))
            
            # Second feature block
            x = self.depthwise(x)
            x = self.pool2(self.bn2(self.act(self.pointwise(x))))
            
            # Dimensionality collapse
            x = self.gap(x)
            x = x.flatten(1)
            
            return self.fc(x)

    return TinyNet()