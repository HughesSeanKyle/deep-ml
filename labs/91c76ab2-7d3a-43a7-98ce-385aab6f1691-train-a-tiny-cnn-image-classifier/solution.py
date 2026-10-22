# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim


class TinyCNN(nn.Module):
    """Small CNN: Conv2d -> ReLU -> pool -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        # Input shape: (N, 1, img_size, img_size) where img_size = 8
        
        # 1. Convolutional stage: Extract 8 distinct spatial filters
        # Using padding=1 on a 3x3 kernel preserves spatial height and width
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # 2. Pooling stage: Shrinks spatial resolution by half (8x8 -> 4x4)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # 3. Calculation of flattened dimensions: 8 channels * 4 height * 4 width = 128
        pooled_size = img_size // 2
        flattened_features = 8 * pooled_size * pooled_size
        
        # 4. Classification stage: Linear mapping to class logits
        self.fc = nn.Linear(flattened_features, n_classes)

    def forward(self, x):
        # Forward execution flow: Conv -> ReLU -> MaxPool
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten the 3D volume into a 1D feature row per batch entry
        # start_dim=1 leaves the batch dimension (N) completely alone
        x = x.flatten(start_dim=1)
        
        # Compute final un-activated class logits
        logits = self.fc(x)
        return logits


def build_model(img_size=8, n_classes=2):
    """Return an instance of your TinyCNN (or equivalent nn.Module)."""
    return TinyCNN(img_size=img_size, n_classes=n_classes)


def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    """Train model on train_x/train_y and return the trained model on CPU."""
    # Enforce exact determinism for reproducibility
    torch.manual_seed(seed)
    
    # Establish optimization criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    num_samples = train_x.size(0)
    
    # Activate training behavior across layers
    model.train()
    
    for epoch in range(epochs):
        # Shuffle indices each epoch to keep gradient updates stochastically healthy
        shuffled_indices = torch.randperm(num_samples)
        
        # Process the full dataset in mini-batch chunks
        for i in range(0, num_samples, batch_size):
            batch_indices = shuffled_indices[i : i + batch_size]
            
            x_batch = train_x[batch_indices]
            y_batch = train_y[batch_indices]
            
            # --- Standard 4-Step PyTorch Optimisation Loop ---
            optimizer.zero_grad()               # 1. Wipe out accumulated gradients
            
            logits = model(x_batch)             # 2. Forward pass evaluation
            loss = criterion(logits, y_batch)   # 3. Scalar loss evaluation
            
            loss.backward()                     # 4. Backward pass error propagation
            optimizer.step()                    # 5. Step weights down the gradient curve
            
    return model