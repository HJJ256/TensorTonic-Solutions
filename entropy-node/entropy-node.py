import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    num_samples = len(y)
    y_unique, counts = np.unique(y, return_counts=True)
    probs = counts/num_samples
    return -np.sum(probs*np.log2(probs))