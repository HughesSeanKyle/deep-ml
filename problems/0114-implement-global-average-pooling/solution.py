import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    """
    Performs Global Average Pooling on a 3D NumPy array.
    
    Args:
        x: Input array of shape (height, width, channels)
        
    Returns:
        A 1D array of shape (channels,) containing the average of each feature map.
    """
    # Compute the mean across both spatial axes (axis 0 = height, axis 1 = width)
    # This leaves only the channels dimension intact.
    return np.mean(x, axis=(0, 1))
