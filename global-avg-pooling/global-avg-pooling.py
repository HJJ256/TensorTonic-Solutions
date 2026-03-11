import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x= np.array(x)
    if len(x.shape) not in [3,4]:
        raise ValueError
    gap_x = x.mean(axis=-1).mean(axis=-1)
    return gap_x