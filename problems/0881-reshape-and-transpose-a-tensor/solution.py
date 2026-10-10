import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    """
    Collapses a tensor down to 1-D and redistributes its elements into a new shape.
    """
    # Using .reshape(-1) completely flattens the tensor regardless of original dimensions.
    # Then we immediately reshape it to the requested new structural shape.
    return x.reshape(-1).reshape(new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    """
    Swaps the final two dimensions of a tensor, preserving any leading batch dimensions.
    """
    # -1 represents the last dimension, -2 represents the second-to-last dimension.
    return x.transpose(-2, -1)
