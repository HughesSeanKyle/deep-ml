import numpy as np

def calculate_auc(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    # 1. Handle edge cases where all samples belong to a single class
    if total_positives == 0 or total_negatives == 0:
        return 0.0
        
    # 2. Replicate the ROC Curve coordinate generation
    unique_scores = np.sort(np.unique(y_scores))[::-1]
    thresholds = [float('inf')] + unique_scores.tolist()
    
    fpr_list = []
    tpr_list = []
    
    for t in thresholds:
        predicted_positives = (y_scores >= t)
        tp = np.sum((y_true == 1) & predicted_positives)
        fp = np.sum((y_true == 0) & predicted_positives)
        
        tpr = tp / total_positives
        fpr = fp / total_negatives
        
        fpr_list.append(float(fpr))
        tpr_list.append(float(tpr))
        
    # 3. Trapezoidal Integration to find Area Under Curve
    auc = 0.0
    for i in range(len(fpr_list) - 1):
        # Calculate the horizontal step size (width)
        width = fpr_list[i+1] - fpr_list[i]
        # Calculate the average vertical height of this step
        avg_height = (tpr_list[i+1] + tpr_list[i]) / 2.0
        # Accumulate the slice area
        auc += width * avg_height
        
    return round(float(auc), 4)