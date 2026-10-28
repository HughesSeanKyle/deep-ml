import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create learnable parameters as nn.Parameter elements
        self.W_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        # Initialise weights using standard PyTorch uniform bounds
        self.reset_parameters()

    def reset_parameters(self):
        """Standard PyTorch initialisation for LSTM cell weights."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, state):
        h_prev, c_prev = state
        
        # 1. Compute packed pre-activations for all 4 gates
        # Shape of g: (N, 4 * hidden_size)
        g = x @ self.W_ih.t() + self.b_ih + h_prev @ self.W_hh.t() + self.b_hh
        
        # 2. Chunk the pre-activations along dim=1 into 4 equal gate tensors
        # Each chunk has a shape of (N, hidden_size)
        i_pre, f_pre, c_tilde_pre, o_pre = torch.chunk(g, chunks=4, dim=1)
        
        # 3. Apply non-linear activation functions
        i = torch.sigmoid(i_pre)
        f = torch.sigmoid(f_pre)
        c_tilde = torch.tanh(c_tilde_pre)
        o = torch.sigmoid(o_pre)
        
        # 4. State updates via element-wise operations (Hadamard product)
        c_new = f * c_prev + i * c_tilde
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new