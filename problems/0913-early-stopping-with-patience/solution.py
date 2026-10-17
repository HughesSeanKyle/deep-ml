import math

class EarlyStopping:
    def __init__(self, patience: int, mode: str = 'min'):
        self.patience = patience
        self.mode = mode
        self.bad_steps = 0
        
        # Initialize boundaries based on the optimization goal
        if self.mode == 'min':
            self.best = math.inf
        elif self.mode == 'max':
            self.best = -math.inf
        else:
            raise ValueError("Mode must be either 'min' or 'max'")

    def step(self, metric: float) -> bool:
        # Determine if a strict improvement occurred
        improved = False
        if self.mode == 'min':
            if metric < self.best:
                improved = True
        elif self.mode == 'max':
            if metric > self.best:
                improved = True
                
        # Action based on improvement status
        if improved:
            self.best = metric
            self.bad_steps = 0
        else:
            self.bad_steps += 1
            
        # Return True if we've run out of patience, False otherwise
        return self.bad_steps >= self.patience
