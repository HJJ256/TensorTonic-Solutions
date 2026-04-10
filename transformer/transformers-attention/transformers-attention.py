import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = K.shape[-1]
    S = torch.einsum("bnd,bmd->bnm",Q,K)/math.sqrt(d_k)
    alpha = F.softmax(S,dim=-1) #For each batch and for each query, compute along all key positions
    out = torch.einsum("bnm,bmd->bnd",alpha,V)
    return out
    