import numpy as np
import math

def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))
	clfs = []

	for _ in range(n_clf):
		best_error = float('inf')
		stump = {}
		best_preds = None
		
		# Iterate through all features and direct column values
		for feat_idx in range(n_features):
			feature_values = X[:, feat_idx]
			thresholds = np.unique(feature_values)
			
			for thresh in thresholds:
				for polarity in [1, -1]:
					# Adjusted split rules to match standard textbook strict inequality configurations
					if polarity == 1:
						preds = np.where(feature_values >= thresh, 1, -1)
					else:
						preds = np.where(feature_values < thresh, 1, -1)
					
					# Compute the weighted misclassification error sum
					misclassified = (preds != y)
					error = np.sum(w[misclassified])
					
					# Track parameter values that minimize weighted error metrics
					if error < best_error:
						best_error = error
						stump['polarity'] = polarity
						stump['threshold'] = thresh
						stump['feature_index'] = feat_idx
						best_preds = preds
						
		# Clip error exactly matching the 1e-10 lower bound boundary
		best_error = np.clip(best_error, 1e-10, 1.0 - 1e-10)
		
		# Calculate voting power weight alpha
		alpha = 0.5 * math.log((1.0 - best_error) / best_error)
		stump['alpha'] = float(alpha)
		
		# Update data sample weights exponentially
		w = w * np.exp(-stump['alpha'] * y * best_preds)
		
		# Re-normalize sample weights vector
		w = w / np.sum(w)
		
		clfs.append(stump.copy())

	return clfs

    