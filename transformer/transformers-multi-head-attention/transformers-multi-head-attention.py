import numpy as np
import math

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    b, s, d = Q.shape
    head_dim = d // num_heads
    Qn = Q @ W_q
    Kn = K @ W_k
    Vn = V @ W_v
    
    Q_new = Qn.reshape(b, s, num_heads, head_dim).transpose(0,2,1,3)
    K_new = Kn.reshape(b, s, num_heads, head_dim).transpose(0,2,1,3)
    V_new = Vn.reshape(b, s, num_heads, head_dim).transpose(0,2,1,3)

    Attention = softmax(Q_new @ K_new.swapaxes(-2,-1) / math.sqrt(head_dim)) @ V_new
    
    
    return Attention.transpose(0,2,1,3).reshape(b,s,d) @ W_o

     
