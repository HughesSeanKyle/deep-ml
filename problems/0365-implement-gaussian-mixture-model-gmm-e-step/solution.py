import numpy as np

def gmm_e_step(X: np.ndarray, means: np.ndarray, variances: np.ndarray, 
               mixing_coeffs: np.ndarray) -> np.ndarray:
    # 1. Cast incoming objects to robust numpy arrays
    X_arr = np.asarray(X, dtype=float)            # Shape: (n_samples,)
    means_arr = np.asarray(means, dtype=float)    # Shape: (n_components,)
    vars_arr = np.asarray(variances, dtype=float)  # Shape: (n_components,)
    priors = np.asarray(mixing_coeffs, dtype=float) # Shape: (n_components,)
    
    # 2. Leverage NumPy broadcasting to map variables simultaneously
    # X_arr[:, np.newaxis] reshapes data points to (n_samples, 1)
    # means_arr[np.newaxis, :] reshapes component parameters to (1, n_components)
    # This results in an evaluation matrix grid of shape: (n_samples, n_components)
    squared_diffs = (X_arr[:, np.newaxis] - means_arr[np.newaxis, :]) ** 2
    
    # 3. Compute univariate Gaussian density matrix components
    gaussian_pdf = (1.0 / np.sqrt(2.0 * np.pi * vars_arr[np.newaxis, :])) * \
                   np.exp(-squared_diffs / (2.0 * vars_arr[np.newaxis, :]))
                   
    # 4. Multiply by mixing coefficients (priors) to get the joint likelihood scores
    weighted_scores = priors[np.newaxis, :] * gaussian_pdf
    
    # 5. Sum across components (axis=1) to compute the normalization denominator vector
    row_sums = np.sum(weighted_scores, axis=1, keepdims=True)
    
    # Inject a tiny epsilon value to protect against accidental division-by-zero on extreme outliers
    row_sums[row_sums == 0.0] = 1e-15
    
    # 6. Compute posterior probabilities (soft assignment responsibility matrix)
    responsibilities = weighted_scores / row_sums
    
    return responsibilities