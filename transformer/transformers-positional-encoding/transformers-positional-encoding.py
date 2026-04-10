import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    arr_pos = np.arange(seq_length).reshape(-1,1) #(seq_length,1)
    arr_i = np.arange(d_model//2).reshape(1,-1) #(1,d_model//2)
    arr_w = 1/(10000)**(2*arr_i/np.sqrt(d_model))
    
    even_pos_enc = np.sin(arr_pos.dot(arr_w)) #(seq_length, d_model//2)
    odd_pos_enc = np.cos(arr_pos.dot(arr_w)) #(seq_length, d_model//2)
    pos_enc = np.empty((seq_length, d_model), dtype=even_pos_enc.dtype)
    pos_enc[:,::2] = even_pos_enc
    pos_enc[:,1::2] = odd_pos_enc
    return pos_enc
    
    