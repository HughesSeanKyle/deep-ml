import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Count overall positive and negative class counts
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    # 1. Extract unique threshold scores sorted in descending order
    unique_scores = np.sort(np.unique(y_scores))[::-1]
    
    # Prepend infinity to represent the state where nothing is classified as positive
    thresholds = [float('inf')] + unique_scores.tolist()
    
    fpr_list = []
    tpr_list = []
    
    # 2. Sweep thresholds from top to bottom
    for t in thresholds:
        # Binary prediction mask vector
        predicted_positives = (y_scores >= t)
        
        # Calculate matrix intersections
        tp = np.sum((y_true == 1) & predicted_positives)
        fp = np.sum((y_true == 0) & predicted_positives)
        
        # 3. Calculate Rates with division-by-zero safeguards
        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0
        
        # Round calculations to 4 decimal places for presentation alignment
        fpr_list.append(round(float(fpr), 4))
        tpr_list.append(round(float(tpr), 4))
        
    return fpr_list, tpr_list