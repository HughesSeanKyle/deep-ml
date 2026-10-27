import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        CRITICAL: Keep this draw order intact to avoid seeding mismatches.
        """
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # Internal cache structures to support BPTT sequence unwinding
        self.x_cache = None
        self.h_cache = None
        self.y_hat_cache = None

    def forward(self, x):
        """
        Forward pass through the RNN for a given sequence of inputs.
        
        Args:
            x: NumPy array of shape (T, input_size)
            
        Returns:
            y_hat: NumPy array of shape (T, output_size, 1)
        """
        T = x.shape[0]
        
        # Pre-allocate cache lists to cleanly store step histories
        # h_cache includes h_0 filled with zeros as its first element
        self.x_cache = []
        self.h_cache = [np.zeros((self.hidden_size, 1))]
        self.y_hat_cache = []
        
        for t in range(T):
            # Reshape input row vector into an explicit column vector (input_size, 1)
            x_t = x[t].reshape(-1, 1)
            self.x_cache.append(x_t)
            
            # Fetch previous time-step hidden state memory
            h_prev = self.h_cache[-1]
            
            # Update current hidden state vector: tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
            z_t = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
            h_t = np.tanh(z_t)
            self.h_cache.append(h_t)
            
            # Compute current time output prediction: W_hy @ h_t + b_y
            y_hat_t = np.dot(self.W_hy, h_t) + self.b_y
            self.y_hat_cache.append(y_hat_t)
            
        # Return stacked array structured to shape (T, output_size, 1)
        return np.array(self.y_hat_cache)

    def backward(self, x, y, learning_rate):
        """
        Backpropagation through time (BPTT) to adjust weights based on error gradient.
        
        Args:
            x: NumPy array of shape (T, input_size)
            y: NumPy array of shape (T, output_size)
            learning_rate: Scalar factor for gradient descent steps
        """
        T = x.shape[0]
        
        # Initialize zero arrays matching original shapes to collect step updates
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # Incoming gradient flowing back from the unrolled future time-step (dz_{t+1})
        dz_next = np.zeros((self.hidden_size, 1))
        
        # Iterate backwards through time axis: from T-1 down to 0
        for t in reversed(range(T)):
            # Reshape actual targets to explicit column vector format
            y_t = y[t].reshape(-1, 1)
            y_hat_t = self.y_hat_cache[t]
            h_t = self.h_cache[t + 1]       # h_cache has offset indices due to h_0
            h_prev = self.h_cache[t]
            x_t = self.x_cache[t]
            
            # 1. Output error derivative computation
            dy_t = y_hat_t - y_t
            
            # 2. Accumulate output parameter adjustments
            dW_hy += np.dot(dy_t, h_t.T)
            db_y += dy_t
            
            # 3. Hidden error routing combining linear layer error and next step recurrent error
            dh_t = np.dot(self.W_hy.T, dy_t) + np.dot(self.W_hh.T, dz_next)
            
            # 4. Derivative map across tanh boundary: dz = dh * (1 - h^2)
            dz_t = dh_t * (1.0 - h_t ** 2)
            
            # 5. Accumulate hidden layer structural modifications
            dW_xh += np.dot(dz_t, x_t.T)
            dW_hh += np.dot(dz_t, h_prev.T)
            db_h += dz_t
            
            # Carry backward error step forward to next sequence iteration loop
            dz_next = dz_t
            
        # Execute vanilla gradient descent parameter position modifications
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y