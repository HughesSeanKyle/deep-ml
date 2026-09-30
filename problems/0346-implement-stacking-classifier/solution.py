import numpy as np

def stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        base_classifiers: list, meta_classifier, n_folds: int = 5) -> np.ndarray:
    # 1. Standardise array types
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    n_samples = X_train.shape[0]
    n_base_clfs = len(base_classifiers)
    m_samples = X_test.shape[0]
    
    # Initialize the out-of-fold feature tracking matrix
    OOF_train = np.zeros((n_samples, n_base_clfs), dtype=int)
    
    # 2. Determine contiguous fold partition boundaries
    fold_size = n_samples // n_folds
    fold_indices = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        # The final fold absorbs any remainder elements left over from integer division
        end_idx = n_samples if i == (n_folds - 1) else (i + 1) * fold_size
        fold_indices.append((start_idx, end_idx))
        
    # 3. Generate Out-of-Fold Meta-Features
    for fold_idx, (val_start, val_end) in enumerate(fold_indices):
        # Create boolean masks to cleanly isolate the training and validation splits
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_start:val_end] = True
        train_mask = ~val_mask
        
        X_tr_fold, y_tr_fold = X_train[train_mask], y_train[train_mask]
        X_val_fold = X_train[val_mask]
        
        # Train each base classifier on this fold's training split and predict on the validation block
        for clf_idx, clf_func in enumerate(base_classifiers):
            val_preds = clf_func(X_tr_fold, y_tr_fold, X_val_fold)
            OOF_train[val_start:val_end, clf_idx] = val_preds
            
    # 4. Generate Meta-Features for the Test Data
    # Base models are trained on the *full* training set to maximise test set performance
    OOF_test = np.zeros((m_samples, n_base_clfs), dtype=int)
    for clf_idx, clf_func in enumerate(base_classifiers):
        test_preds = clf_func(X_train, y_train, X_test)
        OOF_test[:, clf_idx] = test_preds
        
    # 5. Fit the Meta-Classifier on the out-of-fold training data and predict on the test data
    final_predictions = meta_classifier(OOF_train, y_train, OOF_test)
    
    return np.asarray(final_predictions, dtype=int)