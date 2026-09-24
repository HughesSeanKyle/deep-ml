import numpy as np

class Node:
    """Lightweight container for tracking tree structure."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of feature split
        self.threshold = threshold      # Threshold value for split
        self.left = left                # Left child node reference
        self.right = right              # Right child node reference
        self.value = value              # Class prediction label if leaf node

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        # Secure an isolated RandomState for this specific tree execution instance
        self.rng = np.random.RandomState(self.random_state)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _get_gini(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.bincount(y)
        probabilities = counts / m
        return 1.0 - np.sum(probabilities ** 2)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # 1. Evaluate termination/stopping parameters
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            num_labels == 1):
            leaf_val = np.bincount(y).argmax() if len(y) > 0 else 0
            return Node(value=leaf_val)

        # 2. Compute dynamic subset count for max_features
        if self.max_features == 'sqrt':
            max_f = int(np.sqrt(n_features))
        elif self.max_features is None:
            max_f = n_features
        elif isinstance(self.max_features, int):
            max_f = min(self.max_features, n_features)
        else:
            max_f = n_features

        # Select feature subset deterministically via seeded generator
        feature_indices = self.rng.choice(n_features, max_f, replace=False)

        # 3. Find optimal split setup
        best_gini = 1.0
        best_feat, best_thresh = None, None

        for feat in feature_indices:
            X_column = X[:, feat]
            unique_vals = np.unique(X_column)
            
            # Optimization: Cap threshold candidates to prevent timeouts
            if len(unique_vals) > 20:
                thresholds = self.rng.choice(unique_vals, size=20, replace=False)
            else:
                thresholds = unique_vals

            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask
                
                y_left, y_right = y[left_mask], y[right_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate split gini impurity
                g_left = self._get_gini(y_left)
                g_right = self._get_gini(y_right)
                g_split = (len(y_left) / n_samples) * g_left + (len(y_right) / n_samples) * g_right

                if g_split < best_gini:
                    best_gini = g_split
                    best_feat = feat
                    best_thresh = thresh

        # 4. Handle edge cases where no impurity reduction occurs
        if best_feat is None:
            return Node(value=np.bincount(y).argmax())

        # 5. Split and build left/right subtrees recursively
        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def predict(self, X):
        return np.array([self._traverse(inputs, self.root) for inputs in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
