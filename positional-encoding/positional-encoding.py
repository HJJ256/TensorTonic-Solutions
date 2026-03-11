import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos_wi = np.array([[(pos/(base**(2*(i//2)/d_model))) for i in range(d_model)] for pos in range(seq_len)])
    pos_wi[:,::2] = np.sin(pos_wi[:,::2])
    pos_wi[:,1::2] = np.cos(pos_wi[:,1::2])
    return pos_wi
    # Write code here
    