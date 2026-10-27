import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    """
    Process a sequence of input vectors through a simple RNN cell.
    
    Args:
        input_sequence (list or ndarray): Shape (sequence_length, input_dim)
        initial_hidden_state (list or ndarray): Shape (hidden_dim,)
        Wx (list or ndarray): Input-to-hidden weights of shape (hidden_dim, input_dim)
        Wh (list or ndarray): Hidden-to-hidden weights of shape (hidden_dim, hidden_dim)
        b (list or ndarray): Bias vector of shape (hidden_dim,)
        
    Returns:
        list: Final hidden state rounded to four decimal places.
    """
    # Convert arrays to float64 NumPy matrices for rigorous matrix algebra
    h = np.array(initial_hidden_state, dtype=np.float64)
    Wx = np.array(Wx, dtype=np.float64)
    Wh = np.array(Wh, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    # Iterate step-by-step through the input sequence over time axis
    for x_t in input_sequence:
        x_t = np.array(x_t, dtype=np.float64)
        
        # Calculate linear transformation: Wx @ xt + Wh @ h_prev + b
        linear_combination = np.dot(Wx, x_t) + np.dot(Wh, h) + b
        
        # Apply hyperbolic tangent non-linear activation map
        h = np.tanh(linear_combination)
        
    # Round to 4 decimal places and convert back to flat list format
    return np.round(h, 4).tolist()