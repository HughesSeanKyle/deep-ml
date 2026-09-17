from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Initialize optimization tracking tracking states
    best_loss = val_losses[0]
    best_epoch = 0
    patience_counter = 0
    
    # Track through the training history starting from epoch 1
    for epoch in range(1, len(val_losses)):
        current_loss = val_losses[epoch]
        
        # Check if the drop is strictly better than the threshold requirement
        if current_loss < (best_loss - min_delta):
            best_loss = current_loss
            best_epoch = epoch
            patience_counter = 0  # Significant improvement reset
        else:
            patience_counter += 1  # Increment stagnation counter
            
        # Trigger immediate stopping if patience is exhausted
        if patience_counter >= patience:
            return epoch, best_epoch
            
    # If patience is never fully exhausted, return the last indexed epoch
    return len(val_losses) - 1, best_epoch