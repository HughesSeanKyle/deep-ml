import torch

def forward_bf16(model, x):
    # Determine the device type dynamically from the input tensor
    device_type = x.device.type
    
    # Wrap the forward pass in the device-agnostic autocast context
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(x)
    
    # Cast the result back to float32 at the boundary
    return output.float()
