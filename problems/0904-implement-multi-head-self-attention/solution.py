import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # 1. Enforce that embedding dimensions map perfectly across heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 2. Define four bias-free projection matrices
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x shape: (B, T, d_model)
        # mask shape: (T, T) containing 0 and -inf, or None
        B, T, _ = x.shape
        
        # 3. Project input tensor into Q, K, V spaces
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)  # (B, T, d_model)
        v = self.v_proj(x)  # (B, T, d_model)
        
        # 4. Reshape and transpose to separate heads next to the batch dimension
        # Target shape for each: (B, num_heads, T, d_head)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # 5. Calculate scaled dot-product attention scores
        # k.transpose(-2, -1) creates shape (B, num_heads, d_head, T)
        # Resulting scores shape: (B, num_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        # 6. Apply broadcasting mask if provided
        if mask is not None:
            scores = scores + mask
            
        # 7. Apply Softmax along the last dimension (key sequence dimension)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 8. Compute context output vectors
        # Resulting shape: (B, num_heads, T, d_head)
        context = torch.matmul(attention_weights, v)
        
        # 9. Concat all heads back together:
        # Step A: Transpose back to (B, T, num_heads, d_head)
        # Step B: .contiguous() prepares tensor memory allocation layout before flattening layout
        # Step C: Combine into (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        # 10. Pass through output projection layer
        output = self.out_proj(context)
        
        return output