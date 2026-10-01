import numpy as np
import math

def fit_gmm_1d(X, K, initial_means, initial_variances, initial_weights, n_iterations):
    means = list(initial_means)
    variances = list(initial_variances)
    weights = list(initial_weights)
    N = len(X)
    
    # Epsilon value to prevent division by zero or log of zero errors
    eps = 1e-6
    
    for _ in range(n_iterations):
        # ----------------------------------------------------
        # E-STEP: Compute Responsibilities with Safeguards
        # ----------------------------------------------------
        responsibilities = []
        for i in range(N):
            row = []
            total_density = 0.0
            
            for k in range(K):
                diff = X[i] - means[k]
                
                # Safeguard 1: Ensure variance is strictly positive for PDF calculations
                safe_variance = max(variances[k], eps)
                
                exponent = math.exp(- (diff ** 2) / (2 * safe_variance))
                pdf = (1.0 / math.sqrt(2 * math.pi * safe_variance)) * exponent
                
                joint_prob = weights[k] * pdf
                row.append(joint_prob)
                total_density += joint_prob
            
            # Safeguard 2: Prevent division by zero if total_density is 0.0
            if total_density < eps:
                normalized_row = [1.0 / K] * K  # Fallback: distribute responsibility equally
            else:
                normalized_row = [prob / total_density for prob in row]
                
            responsibilities.append(normalized_row)
            
        # ----------------------------------------------------
        # M-STEP: Update Parameters with Safeguards
        # ----------------------------------------------------
        for k in range(K):
            N_k = sum(responsibilities[i][k] for i in range(N))
            
            # Update mixture weight
            weights[k] = N_k / N
            
            # Safeguard 3: Only update mean and variance if the component has responsibility
            if N_k > eps:
                mean_numerator = sum(responsibilities[i][k] * X[i] for i in range(N))
                means[k] = mean_numerator / N_k
                
                var_numerator = sum(responsibilities[i][k] * ((X[i] - means[k]) ** 2) for i in range(N))
                variances[k] = var_numerator / N_k
            else:
                # Fallback if component is dead: leave means unchanged, reset variance/weight
                variances[k] = 0.0
                weights[k] = 0.0

    return {
        'means': [round(m, 4) for m in means],
        'variances': [round(v, 4) for v in variances],
        'weights': [round(w, 4) for w in weights]
    }