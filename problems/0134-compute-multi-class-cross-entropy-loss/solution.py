import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Step 1: Clip probabilities to avoid log(0) numerical instability errors
    # This bounds values between [1e-15, 1 - 1e-15]
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    
    # Step 2: Compute the element-wise cross-entropy formula
    # true_labels * np.log(predicted_probs) will isolate the log of the correct classes
    # Summing along axis=1 adds up the losses per individual sample row
    sample_losses = -np.sum(true_labels * np.log(predicted_probs), axis=1)
    
    # Step 3: Compute the mean loss over the entire batch
    average_loss = np.mean(sample_losses)
    
    return float(average_loss)