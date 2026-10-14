import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model with mini-batches, shuffling, and validation tracking.
    """
    # Step 1: Initialize the optimizer and loss function criterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    n_samples = X_train.shape[0]
    history = []
    
    # Step 2: Main Epoch Loop (1-indexed tracking)
    for epoch in range(1, epochs + 1):
        # Set module to training state
        model.train()
        
        # Shuffle dataset rows at the beginning of each epoch
        perm = torch.randperm(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        epoch_train_loss = 0.0
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Step 3: Loop over Mini-Batches
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # The Canonical PyTorch Training sequence
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            # Track accumulated batch loss
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / n_batches
        
        # Step 4: Run Evaluation / Validation Pass
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            
            # Calculate accuracy by checking correctly classified rows
            val_preds = val_outputs.argmax(dim=1)
            correct_predictions = (val_preds == y_val).sum().item()
            val_accuracy = correct_predictions / len(y_val)
            
        # Step 5: Append metrics snapshot for the current epoch
        history.append({
            'epoch': epoch,
            'train_loss': float(avg_train_loss),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy)
        })
        
    return history
