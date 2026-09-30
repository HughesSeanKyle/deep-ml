import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
	X_arr = np.asarray(X, dtype=float)
	labels_arr = np.asarray(labels, dtype=int)
	
	n_samples = X_arr.shape[0]
	unique_clusters = np.unique(labels_arr)
	n_clusters = len(unique_clusters)
	
	# 1. Edge Case Handling: 
	# If there is only one cluster or each point is its own cluster, return 0.0
	if n_clusters <= 1 or n_clusters == n_samples:
		return 0.0
		
	# Complete pairwise distance matrix computation using vectorized broadcasting
	# Shape: (n_samples, n_samples)
	diffs = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
	distance_matrix = np.sqrt(np.sum(diffs ** 2, axis=2))
	
	silhouette_coefficients = np.zeros(n_samples)
	
	# 2. Iterate sample-by-sample
	for i in range(n_samples):
		current_label = labels_arr[i]
		
		# Separate masks for internal vs external elements relative to sample i
		same_cluster_mask = (labels_arr == current_label)
		
		# Isolate sample index i from its own cluster mean count
		same_cluster_mask[i] = False 
		
		# 3. Calculate internal cohesion factor a(i)
		if np.sum(same_cluster_mask) == 0:
			a_i = 0.0
		else:
			a_i = np.mean(distance_matrix[i, same_cluster_mask])
			
		# 4. Calculate external separation factor b(i)
		b_i = float('inf')
		
		for cluster in unique_clusters:
			if cluster == current_label:
				continue
			
			other_cluster_mask = (labels_arr == cluster)
			mean_dist_to_other_cluster = np.mean(distance_matrix[i, other_cluster_mask])
			
			# Identify the absolute minimum average distance across external neighborhoods
			if mean_dist_to_other_cluster < b_i:
				b_i = mean_dist_to_other_cluster
				
		# 5. Compute individual silhouette coefficient s(i)
		max_val = max(a_i, b_i)
		if max_val == 0.0:
			silhouette_coefficients[i] = 0.0
		else:
			silhouette_coefficients[i] = (b_i - a_i) / max_val
			
	# 6. Aggregate global mean and round to 4 decimal places
	global_score = np.mean(silhouette_coefficients)
	return round(float(global_score), 4)