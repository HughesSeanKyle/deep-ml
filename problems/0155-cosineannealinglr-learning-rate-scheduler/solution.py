import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr: float, T_max: int, min_lr: float):
        # Store tracking parameters
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, epoch: int) -> float:
        # 1. Compute cosine progress multiplier
        # 1 + cos(pi * epoch / T_max) maps from 2.0 (at start) down to 0.0 (at T_max)
        cos_out = math.cos(math.pi * (epoch / self.T_max))
        
        # 2. Scale values within the min_lr to initial_lr target boundaries
        current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + cos_out)
        
        # 3. Round output to exactly 4 decimal places
        return round(current_lr, 4)
