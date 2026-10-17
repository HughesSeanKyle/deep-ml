import math

def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    # Handle out-of-bound edge cases
    if step < 0 or step > total_steps:
        return 0.0
        
    # Phase 1: Linear Warmup
    if step < warmup_steps:
        if warmup_steps == 0:
            return base_lr
        return base_lr * (step / warmup_steps)
        
    # Phase 2: Cosine Annealing
    else:
        # Total steps remaining for decay
        decay_steps = total_steps - warmup_steps
        
        # If training ends immediately at warmup, return base_lr or 0 safely
        if decay_steps == 0:
            return base_lr
            
        # Calculate progress through decay phase (0.0 to 1.0)
        decay_progress = (step - warmup_steps) / decay_steps
        
        # Apply the half-cosine cycle transformation
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        
        return base_lr * cosine_decay
