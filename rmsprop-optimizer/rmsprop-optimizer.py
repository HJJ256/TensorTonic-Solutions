import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)
    s_new = beta*s + (1-beta)*(g**2)
    w_new = w - (lr*g/(s_new+eps)**0.5)
    # Write code here
    return w_new, s_new