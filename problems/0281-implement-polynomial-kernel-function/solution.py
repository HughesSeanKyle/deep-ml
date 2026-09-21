import numpy as np

def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
    # 1. Ensure arrays are standard numpy vectors
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # 2. Compute the fundamental linear dot product interaction
    dot_product = np.dot(x, y)
    
    # 3. Apply the scaling factor and the independent translation offset
    kernel_base = (gamma * dot_product) + coef0
    
    # 4. Exponentiate to map implicitly to higher dimensions
    kernel_value = kernel_base ** degree
    
    # 5. Return as a clean python float
    return float(kernel_value)