import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    N, C, H, W = x.shape
    
    # 1. Compute output spatial dimensions using Ceil Mode math
    H_out = int(np.ceil((H - kernel_size) / stride)) + 1
    W_out = int(np.ceil((W - kernel_size) / stride)) + 1
    
    # 2. Allocate the destination array matrix
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    
    # 3. Slide through the spatial output map
    for h_i in range(H_out):
        for w_i in range(W_out):
            # Calculate the nominal window boundary targets
            h_start = h_i * stride
            w_start = w_i * stride
            
            # Clamp the window slice indices at the actual matrix edge borders
            # This implements the dynamic 'partial window' evaluation logic
            h_end = min(h_start + kernel_size, H)
            w_end = min(w_start + kernel_size, W)
            
            # 4. Extract the slice region and calculate the maximum along spatial dimensions (axis=(-2, -1))
            window = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, h_i, w_i] = np.max(window, axis=(-2, -1))
            
    return out
