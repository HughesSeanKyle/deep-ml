import numpy as np
def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    # 1. Convert to a highly optimized NumPy array matrix
    pred_matrix = np.array(predictions)
    num_classifiers, n_samples = pred_matrix.shape
    
    final_predictions = []
    
    # 2. Iterate sample by sample (Column-by-Column)
    for sample_idx in range(n_samples):
        column_votes = pred_matrix[:, sample_idx]
        
        # Count the occurrences of each class label token
        counts = np.bincount(column_votes)
        max_votes = np.max(counts)
        
        # Identify all classes that achieved the peak highest vote score
        highest_voted_classes = np.where(counts == max_votes)[0]
        
        # 3. Apply the min-label rule: select the mathematically smallest integer index
        winner_class = int(np.min(highest_voted_classes))
        final_predictions.append(winner_class)
        
    return final_predictions