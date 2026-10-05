import math

def train(X_train, y_train, X_val, y_val, n_classes):
    n_features = 64
    
    # 1. Initialize empty prototype accumulators for each class
    # Each centroid starts as a list of 64 zeros, alongside a counter
    centroids = [[0.0] * n_features for _ in range(n_classes)]
    class_counts = [0] * n_classes
    
    # 2. Accumulate pixel sums per class
    for x, y in zip(X_train, y_train):
        class_counts[y] += 1
        for i in range(n_features):
            centroids[y][i] += x[i]
            
    # 3. Compute final mean centroids and their vector lengths (norms)
    centroid_norms = [0.0] * n_classes
    for c in range(n_classes):
        count = class_counts[c] if class_counts[c] > 0 else 1
        # Convert sum to mean
        for i in range(n_features):
            centroids[c][i] /= count
        
        # Precompute the vector norm for the cosine similarity denominator
        sq_sum = sum(centroids[c][i] ** 2 for i in range(n_features))
        centroid_norms[c] = math.sqrt(sq_sum) if sq_sum > 0 else 1.0

    # 4. Define the inference pipeline
    def predict(X):
        predictions = []
        for x in X:
            # Precompute the test vector's norm
            x_sq_sum = sum(v ** 2 for v in x)
            x_norm = math.sqrt(x_sq_sum) if x_sq_sum > 0 else 1.0
            
            best_class = 0
            best_similarity = -1.0 # Cosine similarity bounds are [-1, 1]
            
            # Compare the test vector to all 10 digit centroids
            for c in range(n_classes):
                # Calculate Dot Product
                dot_product = sum(x[i] * centroids[c][i] for i in range(n_features))
                
                # Compute Cosine Similarity
                similarity = dot_product / (x_norm * centroid_norms[c])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_class = c
                    
            predictions.append(best_class)
        return predictions

    return predict
