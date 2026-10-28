import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def _sigmoid(self, val):
        """Helper method for stable sigmoid calculation."""
        return 1 / (1 + np.exp(-val))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, 
        final hidden state, and final cell state.
        """
        T = x.shape[0]
        H = self.hidden_size
        
        # Track initial states across iterations
        h_t = initial_hidden_state
        c_t = initial_cell_state
        
        # Array to store each hidden state over the time sequence
        outputs = np.zeros((T, H, 1))
        
        for t in range(T):
            # 1. Reshape the input vector for step t into a column vector (D, 1)
            x_t = x[t].reshape(-1, 1)
            
            # 2. Stack hidden state first, then input vector to create z_t of shape (H + D, 1)
            z_t = np.vstack((h_t, x_t))
            
            # 3. Compute gate activations (use matrix dot product @)
            f_t = self._sigmoid(self.Wf @ z_t + self.bf)
            i_t = self._sigmoid(self.Wi @ z_t + self.bi)
            c_tilde_t = np.tanh(self.Wc @ z_t + self.bc)
            o_t = self._sigmoid(self.Wo @ z_t + self.bo)
            
            # 4. Perform element-wise mutations to find cell state and hidden state
            c_t = f_t * c_t + i_t * c_tilde_t
            h_t = o_t * np.tanh(c_t)
            
            # 5. Populate outputs array
            outputs[t] = h_t
            
        return outputs, h_t, c_t