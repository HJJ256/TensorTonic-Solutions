import numpy as np
def calc_gini(y):
    y_unique, y_unique_counts = np.unique(y, return_counts=True)
    G = 1 - np.sum((y_unique_counts/len(y))**2)
    return G
def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    X = np.array(X)
    y= np.array(y)
    G_curr = calc_gini(y)
    max_IG = 0
    best_question = (None, None)
    for col in range(X.shape[1]):
        X_col = X[:, col]
        unique_vals = np.unique(X_col)
        mid_pts = [(unique_vals[i]+unique_vals[i+1])/2 for i in range(len(unique_vals)-1)]
        for mid_pt in mid_pts:
            y_left =  y[X_col < mid_pt]
            y_right =  y[X_col > mid_pt]
            G_left = calc_gini(y_left)
            G_right = calc_gini(y_right)
            IG = G_curr - len(y_left)*G_left/len(y) - len(y_right)*G_right/len(y)
            if IG>max_IG:
                max_IG = IG
                best_question = (col, mid_pt)
    return best_question
            
        
    