import numpy as np

class NaiveBayes():
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.classes = None
        self.class_log_priors = {}
        self.feature_log_prob_1 = {} # log P(x_i = 1 | y)
        self.feature_log_prob_0 = {} # log P(x_i = 0 | y)

    def forward(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model to binary features X and labels y using Laplace smoothing.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # FIX: Iterate explicitly over [0, 1] to ensure both classes are 
        # accounted for, even if one is missing from the training labels.
        for c in [0, 1]:  
            # Mask out records belonging to class c
            X_c = X[y == c]
            count_c = X_c.shape[0]
            
            # 1. Compute Class Log Priors smoothly
            # Handles edge cases where a class has 0 samples during training
            # prior = (count_c + self.smoothing) / (n_samples + 2 * self.smoothing)
            # self.class_log_priors[c] = np.log(prior)
            prior = count_c / n_samples
            self.class_log_priors[c] = np.log(prior) if prior > 0 else -np.inf
            
            # 2. Compute Feature Likelihoods with Laplace Smoothing
            # Sum up occurrences of 1s in each column
            feature_counts = np.sum(X_c, axis=0) if count_c > 0 else np.zeros(n_features)
            
            prob_1 = (feature_counts + self.smoothing) / (count_c + 2 * self.smoothing)
            prob_0 = 1.0 - prob_1
            
            # Cache log states directly to use during inference phase
            self.feature_log_prob_1[c] = np.log(prob_1)
            self.feature_log_prob_0[c] = np.log(prob_0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts binary class labels for a test matrix X.
        """
        n_samples = X.shape[0]
        # Array to store log posteriors for class 0 and class 1
        log_posteriors = np.zeros((n_samples, 2))
        
        # FIX: Iterate explicitly over [0, 1] to match the 2 columns 
        # in our log_posteriors array.
        for c in [0, 1]:
            # Pull parameters from our training cache
            log_prior = self.class_log_priors[c]
            log_p1 = self.feature_log_prob_1[c]
            log_p0 = self.feature_log_prob_0[c]
            
            # Vectorised log-likelihood calculation:
            # For each feature: X * log_p1 + (1 - X) * log_p0
            # Then sum across columns (axis=1) and add log_prior
            likelihood_matrix = X * log_p1 + (1 - X) * log_p0
            log_posteriors[:, c] = log_prior + np.sum(likelihood_matrix, axis=1)
            
        # The predicted label is the column index (0 or 1) with the highest score
        return np.argmax(log_posteriors, axis=1)