import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 1. Extract unique thresholds sorted in descending order
    thresholds = np.sort(np.unique(y_scores))[::-1]
    
    precisions = []
    recalls = []
    
    # Total actual positive counts in the entire dataset
    actual_positives = np.sum(y_true == 1)
    
    # 2. Iterate through each threshold configuration
    for t in thresholds:
        # Binary mask vector of what passes our threshold cutoff
        predicted_positives_mask = (y_scores >= t)
        predicted_positives_count = np.sum(predicted_positives_mask)
        
        # Calculate True Positives
        tp = np.sum((y_true == 1) & predicted_positives_mask)
        
        # 3. Apply Edge Case Rule for Precision
        if predicted_positives_count == 0:
            precision = 1.0
        else:
            precision = tp / predicted_positives_count
            
        # 4. Apply Edge Case Rule for Recall
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = tp / actual_positives
            
        # Save calculations rounded to 4 decimal places for clean formatting
        precisions.append(round(float(precision), 4))
        recalls.append(round(float(recall), 4))
        
    return precisions, recalls, thresholds.tolist()