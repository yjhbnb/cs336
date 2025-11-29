import torch
from building_blocks import *

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        ln1 = RMSNorm(d_model=d_model)
        ln2 = RMSNorm(d_model=d_model)
        ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))
        return x
