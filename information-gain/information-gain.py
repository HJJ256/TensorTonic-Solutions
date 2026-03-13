import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.array(y)
    split_mask = np.array(split_mask)
    y_left = y[split_mask]
    y_right = y[~split_mask]
    P_left = len(y_left)/len(y)
    P_right = len(y_right)/len(y)
    IG = _entropy(y) - P_left*_entropy(y_left) - P_right*_entropy(y_right)
    return IG
