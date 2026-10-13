import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    """
    Serializes the state dict of the source model into an in-memory buffer,
    rewinds the buffer stream, and loads it directly into the destination model.
    """
    # Step 1: Initialize a fresh in-memory byte buffer stream
    buffer = io.BytesIO()
    
    # Step 2: Serialize the source model's state dictionary directly into the RAM stream
    torch.save(src.state_dict(), buffer)
    
    # Step 3: Rewind the buffer's stream pointer back to the beginning (0th byte)
    buffer.seek(0)
    
    # Step 4: Deserialize the byte data back into a valid state dictionary object
    src_state_dict = torch.load(buffer, weights_only=True)
    
    # Step 5: Overwrite the parameter matrices of the destination module in place
    dst.load_state_dict(src_state_dict)
    
    return dst
