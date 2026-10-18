import torch
import torch.nn as nn


class RegularizedMLP(nn.Module):
    """MLP with BatchNorm1d and Dropout for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.3):
        super().__init__()
        
        # Constructing 2 identical regularized hidden blocks sequentially
        # Block Order: Linear -> BatchNorm1d -> Activation -> Dropout
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            # Block 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            # Final output layer mapping to a single raw logit unit
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) logits for batch x of shape (N, input_dim)."""
        logits = self.network(x)
        # Squeeze the final dimension from shape (N, 1) down to (N,)
        return logits.squeeze(-1)


def train_model(model, X_train, y_train, epochs=150, lr=1e-2):
    """Train model in-place with BCEWithLogitsLoss + Adam. Return model."""
    # Ensure the model is set explicitly to training mode to activate BatchNorm & Dropout
    model.train()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # 1. Reset stored parameter gradients
        optimizer.zero_grad()
        
        # 2. Forward pass to acquire predictions
        logits = model(X_train)
        
        # 3. Calculate binary cross-entropy loss against float targets
        loss = criterion(logits, y_train)
        
        # 4. Backward pass to propagate errors and calculate gradients
        loss.backward()
        
        # 5. Advance model weights using the Adam optimizer
        optimizer.step()
        
    return model
