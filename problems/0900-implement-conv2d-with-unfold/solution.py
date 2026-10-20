import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    # 1. Extract shapes (using unique names to avoid overwriting the W tensor!)
    N, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = W.shape
    
    # 2. Compute spatial dimensions of the output
    H_out = ((H_in + 2 * padding - kH) // stride) + 1
    W_out = ((W_in + 2 * padding - kW) // stride) + 1
    
    # 3. Extract sliding window patches
    x_unfolded = F.unfold(x, kernel_size=(kH, kW), stride=stride, padding=padding)
    
    # 4. Flatten the weight tensor
    W_flat = W.view(C_out, -1)
    
    # 5. Perform matrix multiplication
    out_flat = torch.matmul(W_flat, x_unfolded)
    
    # 6. Reshape to standard image structure
    out = out_flat.view(N, C_out, H_out, W_out)
    
    # 7. Add bias if present
    if b is not None:
        out += b.view(1, C_out, 1, 1)
        
    return out
