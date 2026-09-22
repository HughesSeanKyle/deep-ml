from collections import Counter
def balance_undersample(data: list) -> list:
    # 1. Handle empty input edge case
    if not data:
        return []
    
    # 2. Determine the minimum count across all class labels
    # Extract only the labels to find frequencies
    labels = [label for _, label in data]
    counts = Counter(labels)
    min_count = min(counts.values())
    
    # 3. Initialize state tracking dictionary for kept items
    kept_counts = Counter()
    balanced_data = []
    
    # 4. Perform a single-pass filter over the original dataset
    for sample, label in data:
        if kept_counts[label] < min_count:
            balanced_data.append((sample, label))
            kept_counts[label] += 1
            
    return balanced_data
