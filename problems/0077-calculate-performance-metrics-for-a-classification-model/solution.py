
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    # Collect core metrics counts
    for act, pred in zip(actual, predicted):
        if act == 1 and pred == 1:
            tp += 1
        elif act == 0 and pred == 0:
            tn += 1
        elif act == 0 and pred == 1:
            fp += 1
        elif act == 1 and pred == 0:
            fn += 1
            
    # Standard format where Rows = Actual, Columns = Predicted
    # [[TP, FN], 
    #  [FP, TN]]
    confusion_matrix = [[tp, fn], [fp, tn]]
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Calculate performance metrics with safe zero-division fallbacks
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    negativePredictive = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
