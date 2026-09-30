import numpy as np

def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
	# 1. Establish the random state at the start if provided
	if seed is not None:
		np.random.seed(seed)
		
	n_samples, n_features = X.shape
	centroids = np.zeros((k, n_features))
	
	# 2. Requirement: Use np.random.randint for the first centroid selection
	first_idx = np.random.randint(0, n_samples)
	centroids[0] = X[first_idx]
	
	# 3. Iteratively select the remaining k-1 centroids
	for c_idx in range(1, k):
		# Matrix tracking squared distances from all points to all currently chosen centroids
		# Shape: (n_samples, current_num_centroids)
		active_centroids = centroids[:c_idx]
		
		# Efficient broadcasting to compute squared differences: (n_samples, c_idx, n_features)
		# We look across features (axis=2) to get the squared Euclidean distance
		sq_distances = np.sum((X[:, np.newaxis, :] - active_centroids[np.newaxis, :, :]) ** 2, axis=2)
		
		# Isolate the shortest distance to *any* already-chosen centroid
		min_sq_distances = np.min(sq_distances, axis=1)
		
		# 4. Compute selection probabilities proportional to squared distance
		sum_sq_distances = np.sum(min_sq_distances)
		
		# Safeguard against division by zero if all unchosen points sit exactly on existing centroids
		if sum_sq_distances == 0.0:
			probs = np.ones(n_samples) / n_samples
		else:
			probs = min_sq_distances / sum_sq_distances
			
		# 5. Requirement: Use np.random.choice with the computed probabilities
		next_idx = np.random.choice(n_samples, p=probs)
		centroids[c_idx] = X[next_idx]
		
	return centroids