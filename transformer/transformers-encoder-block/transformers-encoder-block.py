import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mu = np.mean(x, axis=-1, keepdims=True)
    sigma_2 = np.mean((x-mu)**2, axis=-1, keepdims=True)
    out = gamma*(x-mu)/np.sqrt(sigma_2+eps) + beta
    return out

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B,L,D = K.shape
    d_h = D//num_heads
    Q = np.dot(Q,W_q).reshape(B,L,num_heads, d_h)
    Q = Q.transpose(0,2,1,3)
    K = np.dot(K,W_k).reshape(B,L,num_heads, d_h)
    K = K.transpose(0,2,3,1)
    V = np.dot(V,W_v).reshape(B,L,num_heads, d_h)
    V = V.transpose(0,2,1,3)

    alpha = softmax(np.matmul(Q,K)/np.sqrt(d_h))
    out = np.matmul(alpha, V)
    out = out.transpose(0,2,1,3).reshape(B,L,D)
    return np.dot(out,W_o)
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    h = np.dot(x, W1) + b1
    h_p = np.maximum(0,h)
    out = np.dot(h_p, W2) + b2
    return out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x1 = x + attn
    x_p = layer_norm(x1, gamma1, beta1)
    ffn = feed_forward(x_p, W1, b1, W2, b2)
    x2 = x_p + ffn
    out = layer_norm(x2, gamma2, beta2)
    return out