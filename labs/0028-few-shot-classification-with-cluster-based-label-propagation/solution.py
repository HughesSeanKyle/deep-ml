import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

def train(X_unlabeled, X_labeled, y_labeled, X_val, y_val):
    # Set K much larger than 10 to ensure dense, highly pure sub-clusters
    K = 60 
    
    # 1. Combine all available structural data 
    X_all = np.vstack([X_labeled, X_unlabeled])
    n_labeled = len(X_labeled)

    # 2. Compute the cluster geometries
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    cluster_assignments = km.fit_predict(X_all)

    # 3. Map cluster IDs to labeled ground truth
    cluster_to_label_map = {}
    for k in range(K):
        # Check which labeled points landed in cluster k
        labeled_in_cluster = (cluster_assignments[:n_labeled] == k)
        
        if labeled_in_cluster.any():
            # If multiple labels land in the same tight cluster, take the majority vote
            majority_label = np.bincount(y_labeled[labeled_in_cluster]).argmax()
            cluster_to_label_map[k] = int(majority_label)

    # 4. Filter and build the high-purity pseudo-labeled dataset
    keep_indices = [i for i in range(len(X_all)) if cluster_assignments[i] in cluster_to_label_map]
    
    X_pseudo = X_all[keep_indices]
    y_pseudo = np.array([cluster_to_label_map[cluster_assignments[i]] for i in keep_indices])

    # 5. Train downstream classifier on the expanded, highly reliable dataset
    # Mild L2 regularization (C=1.0) generalizes well on the smooth pseudo-labels
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    clf.fit(X_pseudo, y_pseudo)

    # 6. Return the isolated predictive callable
    def predict(X):
        return clf.predict(X)

    return predict
