import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    # 1. Zero out gradients once at the start
    optimizer.zero_grad()
    
    total_unscaled_loss = 0.0
    num_batches = len(micro_batches)
    
    for x, y in micro_batches:
        # 2. Forward pass for the micro-batch
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Track the raw, unscaled loss value
        total_unscaled_loss += loss.item()
        
        # 3. Scale the loss by 1/N and accumulate gradients
        scaled_loss = loss / num_batches
        scaled_loss.backward()
        
    # 4. Perform a single optimizer step using the aggregated gradients
    optimizer.step()
    
    # 5. Return the true mean loss across all micro-batches
    return total_unscaled_loss / num_batches
