import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. First normalisation layer before self-attention
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Native PyTorch MultiheadAttention configured for batch-first sequences
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 3. Second normalisation layer before the MLP network
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Multi-Layer Perceptron (Feed-Forward Network) block with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 5. Shared dropout instance applied to sub-layer outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (B, T, d_model)
        
        # --- SUB-LAYER 1: Pre-LN Attention ---
        # A. Normalise input
        normed_x1 = self.norm1(x)
        # B. Run self-attention (passing normed tensor as Q, K, and V)
        attn_out, _ = self.attn(normed_x1, normed_x1, normed_x1)
        # C. Apply dropout and add the original, un-normalised x (residual connection)
        x = x + self.dropout(attn_out)
        
        # --- SUB-LAYER 2: Pre-LN MLP ---
        # A. Normalise intermediate state
        normed_x2 = self.norm2(x)
        # B. Run through Feed-Forward network
        mlp_out = self.mlp(normed_x2)
        # C. Apply dropout and add the un-normalised intermediate x (residual connection)
        x = x + self.dropout(mlp_out)
        
        return x
