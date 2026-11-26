import numpy as np

def sliding_kv_cache_update(cache_k, cache_v, current_len, new_k, new_v, window_size):
    """
    Update a pre-allocated sliding-window KV cache and produce the causal mask
    for the new queries against the updated cache.

    Args:
        cache_k, cache_v: nested lists / arrays of shape
            (batch, num_heads, window_size, head_dim).
        current_len: int, number of valid entries currently in the cache (<= window_size).
        new_k, new_v: nested lists / arrays of shape
            (batch, num_heads, new_len, head_dim).
        window_size: int, maximum cache length.

    Returns:
        dict with keys 'cache_k', 'cache_v', 'current_len', 'mask'.
    """
    # Copy inputs as float arrays so original buffers are never mutated
    out_k = np.array(cache_k, dtype=np.float64, copy=True)
    out_v = np.array(cache_v, dtype=np.float64, copy=True)
    nk = np.asarray(new_k, dtype=np.float64)
    nv = np.asarray(new_v, dtype=np.float64)

    new_len = nk.shape[2]
    updated_len = int(min(current_len + new_len, window_size))

    # Edge case: if incoming chunk alone meets or exceeds window_size,
    # only the trailing `window_size` tokens of new_k/new_v survive
    if new_len >= window_size:
        out_k[:, :, :window_size, :] = nk[:, :, -window_size:, :]
        out_v[:, :, :window_size, :] = nv[:, :, -window_size:, :]
    else:
        total = current_len + new_len
        overflow = max(0, total - window_size)
        keep_old = current_len - overflow

        # Shift surviving old entries to the left (positions 0 .. keep_old - 1)
        if overflow > 0 and keep_old > 0:
            out_k[:, :, :keep_old, :] = out_k[:, :, overflow:current_len, :]
            out_v[:, :, :keep_old, :] = out_v[:, :, overflow:current_len, :]

        # Append all new entries right after the surviving old ones
        out_k[:, :, keep_old:keep_old + new_len, :] = nk
        out_v[:, :, keep_old:keep_old + new_len, :] = nv

    # Construct the (new_len, updated_len) causal mask
    query_positions = np.maximum(updated_len - new_len + np.arange(new_len), 0)
    key_positions = np.arange(updated_len)
    mask = (key_positions[None, :] <= query_positions[:, None]).astype(int)

    return {
        "cache_k": out_k.tolist(),
        "cache_v": out_v.tolist(),
        "current_len": updated_len,
        "mask": mask.tolist(),
    }