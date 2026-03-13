import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    w = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
    return w