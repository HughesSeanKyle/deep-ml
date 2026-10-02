import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    n_samples, n_features = data.shape
    
    # ----------------------------------------------------
    # STEP 1: Precompute the Kernel Matrix (K)
    # ----------------------------------------------------
    K = np.zeros((n_samples, n_samples))
    if kernel == 'linear':
        K = np.dot(data, data.T)
    elif kernel == 'rbf':
        sq_norms = np.sum(data**2, axis=1).reshape(-1, 1)
        dist_matrix = sq_norms + sq_norms.T - 2 * np.dot(data, data.T)
        K = np.exp(-dist_matrix / (2 * (sigma ** 2)))

    alphas = np.zeros(n_samples)
    bias = 0.0

    # ----------------------------------------------------
    # STEP 2: Deterministic Batch Pegasos Loop
    # ----------------------------------------------------
    for t in range(1, iterations + 1):
        eta = 1.0 / (lambda_val * t)
        
        # NO SHUFFLING - iterate in natural order
        for i in range(n_samples):
            # Decision function
            kernel_sum = np.sum(alphas * labels * K[:, i])
            f_xi = kernel_sum + bias
            
            # FIX: Apply decay and update according to Pegasos rule
            if labels[i] * f_xi < 1:
                # Violated: decay + add η_t
                alphas[i] = (1 - eta * lambda_val) * alphas[i] + eta
                bias += eta * labels[i]
            else:
                # Not violated: decay only
                alphas[i] = (1 - eta * lambda_val) * alphas[i]

    return list(alphas), float(bias)