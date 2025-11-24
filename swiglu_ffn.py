import torch
from linear import Linear

class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_ff = self.d_model * 8 // 3

        self.silu_linear   = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)
        self.gate_linear = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)
        self.final_linear  = Linear(in_features=self.d_ff, out_features=self.d_model, bias=False)

    def forward(self, x):
        assert x.shape[-1] == self.d_model, "shape mismatched"

        silu_ffn_x = self.silu_linear(x)                # (..., d_ff)
        silu_x     = silu_ffn_x * torch.sigmoid(silu_ffn_x)  # SiLU(...), (..., d_ff)

        normal_x   = silu_x * self.gate_linear(x)     # gate ⊙ W3 x, (..., d_ff)
        out        = self.final_linear(normal_x)        # (..., d_model)

        return out.to(x.dtype)
