import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    B,L,D = K.shape
    d_h = D//num_heads
    q = np.dot(Q,W_q).reshape(B,L,num_heads,d_h)
    q = q.transpose((0,2,1,3))
    k = np.dot(K,W_k).reshape(B,L,num_heads,d_h)
    k = k.transpose((0,2,3,1))
    v = np.dot(V,W_v).reshape(B,L,num_heads,d_h)
    v = v.transpose((0,2,1,3))
    
    s = np.matmul(q,k)/np.sqrt(d_h)
    alpha = softmax(s)
    out = np.matmul(alpha,v)
    out = out.transpose((0,2,1,3)).reshape((B,L,-1))
    return out