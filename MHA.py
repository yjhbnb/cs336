import torch
from torch import nn
import einops

from linear import Linear
from rope import RotaryPositionalEmbedding
from attention import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        # project into Q, K, V for all heads at once: (B, S, d_model) -> (B, S, h * d_k)
        self.WQ = Linear(in_features=self.d_model, out_features=self.num_heads * self.d_k)
        self.WK = Linear(in_features=self.d_model, out_features=self.num_heads * self.d_k)
        self.WV = Linear(in_features=self.d_model, out_features=self.num_heads * self.d_v)

        # output projection: (B, S, h * d_v) -> (B, S, d_model)
        self.WO = Linear(in_features=self.num_heads * self.d_v, out_features=self.d_model)

        self.theta = theta
        self.max_seq_len = max_seq_len
        self.rope = RotaryPositionalEmbedding(theta=self.theta,
                                              d_k=self.d_k,
                                              max_seq_len=self.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d_model)
        token_positions (optional): (B, S) absolute positions. If None, use 0..S-1.
        """
        B, S, D = x.shape
        assert D == self.d_model, "shape mismatched"

        # positions: [0, 1, ..., S-1] for each sequence in the batch
        token_positions = einops.rearrange(torch.arange(S, device=x.device), "s -> 1 s")

        # 1) Linear projections
        Q = self.WQ(x)  # (B, S, h * d_k)
        K = self.WK(x)  # (B, S, h * d_k)
        V = self.WV(x)  # (B, S, h * d_v)

        # 2) Reshape to heads: (B, S, h * d) -> (B, h, S, d)
        Q = einops.rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = einops.rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = einops.rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # 3) Apply RoPE to Q and K (head dim is treated as batch)
        Q = self.rope(Q, token_positions=token_positions)  # (B, h, S, d_k)
        K = self.rope(K, token_positions=token_positions)

        # 4) Causal mask: allow j <= i
        mask = torch.tril(torch.ones((S, S), dtype=torch.bool, device=x.device))

        # 5) Scaled dot-product attention per head
        att = scaled_dot_product_attention(Q, K, V, mask=mask)  # (B, h, S, d_v)

        # 6) Merge heads back: (B, h, S, d_v) -> (B, S, h * d_v)
        att = einops.rearrange(att, "b h s d -> b s (h d)", h=self.num_heads)

        # 7) Final linear projection
        out = self.WO(att)  # (B, S, d_model)

        return out.to(x.dtype)
