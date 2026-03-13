import numpy as np

def calc_gini(y):
    if len(y) == 0:
        return 0.
    _, unique_counts = np.unique(y, return_counts=True)
    G = 1 - np.sum((unique_counts/len(y))**2)
    return G
    
def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    if len(y_left)==0 and len(y_right)==0:
        return 0.
    G_left = calc_gini(y_left)
    G_right = calc_gini(y_right)
    
    G_split = (len(y_left)*G_left + len(y_right)*G_right)/(len(y_left)+len(y_right))
    return G_split
    