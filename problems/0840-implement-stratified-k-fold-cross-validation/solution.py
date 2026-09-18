import numpy as np

def stratified_kfold_indices(y, n_splits):
    y = np.array(y)
    n_samples = len(y)
    unique_classes = np.unique(y)
    
    # 1. Isolate original-ordered indices for each individual class
    class_indices = {}
    for c in unique_classes:
        class_indices[c] = np.where(y == c)[0]
        
    # Create buckets to store the split groups for each class
    # class_groups[c][i] will hold Group i for Class c
    class_groups = {c: [] for c in unique_classes}
    
    # 2. Partition indices of each class into n_splits consecutive groups
    for c in unique_classes:
        indices = class_indices[c]
        c_size = len(indices)
        
        base_size = c_size // n_splits
        remainder = c_size % n_splits
        
        start_idx = 0
        for i in range(n_splits):
            # Give an extra sample to the first 'remainder' groups
            current_group_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_group_size
            
            # Store the consecutive slice
            class_groups[c].append(indices[start_idx:end_idx])
            start_idx = end_idx

    # 3. Assemble the folds by combining the groups
    folds = []
    for i in range(n_splits):
        test_fold_indices = []
        train_fold_indices = []
        
        for c in unique_classes:
            # Test set gets group i from class c
            test_fold_indices.extend(class_groups[c][i])
            
            # Train set gets all other groups from class c
            for j in range(n_splits):
                if j != i:
                    train_fold_indices.extend(class_groups[c][j])
                    
        # 4. Strictly sort indices in ascending order as requested
        train_fold_indices.sort()
        test_fold_indices.sort()
        
        folds.append([train_fold_indices, test_fold_indices])
        
    return folds