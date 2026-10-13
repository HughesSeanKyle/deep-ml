import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout rate p must be between 0 and 1 (exclusive of 1).")
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer."""
        if training and self.p > 0.0:
            scale_factor = 1.0 / (1.0 - self.p)
            # Threshold using standard (1 - p) comparison logic to match random seed behavior
            self.mask = (np.random.rand(*x.shape) < (1.0 - self.p)) * scale_factor
            return x * self.mask
        else:
            # Crucial: Do NOT reset self.mask to None here, 
            # so the backward pass can still access the training pass mask!
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        if self.mask is not None:
            return grad * self.mask
        return grad