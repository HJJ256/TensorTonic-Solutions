import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    if rng:
        random_arr = rng.random((x.shape))
    else:
        random_arr = np.random.random((x.shape))
    dropout_pattern = np.where(random_arr<(1-p), 1/(1-p), 0)
    return x*dropout_pattern, dropout_pattern