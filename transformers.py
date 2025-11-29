import torch
from building_blocks import *

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        norm1 = RMSNorm(d_model=d_model)
        norm2 = RMSNorm(d_model=d_model)
        swiglu_ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        att = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + att(norm1(x))
        x = x + swiglu_ffn(norm2(x))
        return x
