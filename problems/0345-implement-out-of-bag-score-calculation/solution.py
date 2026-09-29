import numpy as np

def calculate_oob_score(n_samples: int, bootstrap_indices: list, predictions: list, y_true: list) -> float:
    # 1. Cast lists to clean arrays for fast vector slicing
    pred_matrix = np.array(predictions, dtype=int)  # Shape: (n_estimators, n_samples)
    y_true_arr = np.array(y_true, dtype=int)
    n_estimators = len(bootstrap_indices)
    
    # 2. Convert bootstrap lists into optimized unique set lookups per tree
    bootstrap_sets = [set(indices) for indices in bootstrap_indices]
    
    oob_correct_count = 0
    total_evaluated_samples = 0
    
    # 3. Step sample-by-sample (column-by-column)
    for sample_idx in range(n_samples):
        sample_oob_votes = []
        
        for est_idx in range(n_estimators):
            # Check if this sample was excluded from the current tree's training bag
            if sample_idx not in bootstrap_sets[est_idx]:
                sample_oob_votes.append(pred_matrix[est_idx, sample_idx])
                
        # 4. If a sample was in-bag for all estimators, ignore it completely
        if len(sample_oob_votes) == 0:
            continue
            
        # 5. Aggregate predictions using majority vote
        consensus_pred = np.bincount(sample_oob_votes).argmax()
        
        # 6. Track accurate consensus labels
        if consensus_pred == y_true_arr[sample_idx]:
            oob_correct_count += 1
            
        total_evaluated_samples += 1
        
    # 7. Protect against empty sets or extreme edge cases where no rows are OOB
    if total_evaluated_samples == 0:
        return 0.0
        
    return float(oob_correct_count / total_evaluated_samples)