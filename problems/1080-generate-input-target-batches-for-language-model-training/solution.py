import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    """
    Sample random input-target training batches for next-token prediction.

    Args:
        data: 1D numpy array of integer token IDs
        block_size: context length (number of tokens per sequence)
        batch_size: number of sequences in the batch
        seed: integer seed for reproducibility

    Returns:
        numpy array of shape (2, batch_size, block_size) with integer dtype,
        where index 0 is input x and index 1 is target y.
    """
    data = np.asarray(data, dtype=np.int64)
    rng = np.random.default_rng(seed)
    
    # Sample batch_size random starting indices in [0, len(data) - block_size)
    ix = rng.integers(0, len(data) - block_size, size=batch_size)
    
    # Build input sequences x and right-shifted target sequences y
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    
    # Stack x and y along a new leading axis to get shape (2, batch_size, block_size)
    return np.stack([x, y], axis=0)