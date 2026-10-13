import torch
from torch.utils.data import Dataset

class TensorPairDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Initializes the dataset by storing matching input data and label tensors.
        """
        # Ensure dimensions match to protect against downstream shape mismatches
        assert X.shape[0] == y.shape[0], "Features X and Labels y must have identical lengths!"
        
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returns the total number of samples contained within the dataset.
        """
        # The size of the leading dimension (axis 0) represents the dataset length
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single data feature and target label pair at the specified index.
        """
        # Slicing along axis 0 pulls the exact singular row needed
        return self.X[idx], self.y[idx]
