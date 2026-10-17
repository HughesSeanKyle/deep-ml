class StepLRScheduler:
    def __init__(self, initial_lr: float, step_size: int, gamma: float):
        # Store configuration parameters
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        # 1. Determine how many times gamma has been applied
        decay_count = epoch // self.step_size
        
        # 2. Calculate the raw learning rate using exponential decay
        current_lr = self.initial_lr * (self.gamma ** decay_count)
        
        # 3. Round the final answer to 4 decimal places
        return round(current_lr, 4)
