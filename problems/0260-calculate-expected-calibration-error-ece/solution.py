import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    N = len(y_prob)
    
    # Generate the linear boundary points for equal-width spaces
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for m in range(n_bins):
        lower = bin_edges[m]
        upper = bin_edges[m + 1]
        
        # Isolate indices belonging to the current bin partition
        if m == 0:
            # First bin includes both endpoints [0, 1/n_bins]
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            # Subsequent bins exclude lower boundary (lower, upper]
            in_bin = (y_prob > lower) & (y_prob <= upper)
            
        bin_count = np.sum(in_bin)
        
        # Calculate metric score only if the bin contains data points
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            
            # Weigh the absolute error metric by structural population size
            bin_weight = bin_count / N
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return round(float(ece), 3)