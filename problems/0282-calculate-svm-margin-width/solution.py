import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    # Calculate the L2 norm (Euclidean norm) of the weight vector
    w_norm = np.linalg.norm(w)
    
    # The margin width is defined as 2 divided by the L2 norm
    margin_width = 2.0 / w_norm
    
    return float(margin_width)