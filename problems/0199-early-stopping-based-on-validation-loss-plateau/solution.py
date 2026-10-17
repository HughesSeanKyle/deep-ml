def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
    """
    Determine at each epoch whether training should stop based on validation loss.
    
    Args:
        val_losses: List of validation losses at each epoch
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
    
    Returns:
        List of booleans indicating whether to stop at each epoch
    """
    # 1. Immediate edge case check
    if not val_losses:
        return []
        
    stop_signals = []
    best_loss = None
    stagnant_epochs = 0
    
    for loss in val_losses:
        # If this is the very first epoch, establish the initial baseline
        if best_loss is None:
            best_loss = loss
            stagnant_epochs = 0
        # Check if the new loss qualifies as a significant improvement
        elif loss < best_loss - min_delta:
            best_loss = loss
            stagnant_epochs = 0
        # Otherwise, the model is stagnating
        else:
            stagnant_epochs += 1
            
        # Determine if patience has run out
        if stagnant_epochs >= patience:
            stop_signals.append(True)
        else:
            stop_signals.append(False)
            
    return stop_signals
