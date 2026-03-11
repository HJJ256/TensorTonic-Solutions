import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def loss_derivative(y,p,X, N):
    diff_p_y = (p-y) #N,
    return (X.T.dot(diff_p_y))/N, np.average(diff_p_y)
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N,D = X.shape
    # Write code here
    W = np.zeros((D,))
    b = 0.0
    for step in range(steps):
        z = X.dot(W) + b
        p = _sigmoid(z) # N,
        loss = -np.average(y*np.log(p) + (1-y)*np.log(1-p))
        der_w, der_b = loss_derivative(y,p,X,N)
        W = W - lr*der_w
        b = b - lr*der_b
    return W,b