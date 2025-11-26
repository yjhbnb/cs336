import torch
import einops

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Implements RoPE (Su et al., 2021).
    Rotates Q/K vectors in 2D slices of size 2.

    Inputs:
        x:              (..., seq_len, d_k)
        token_positions:(..., seq_len)

    Returns:
        x_rotated:      (..., seq_len, d_k)
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.d_k = d_k                      # hidden dim of each head
        self.max_seq_len = max_seq_len

        cos, sin = self._cache_rope_freq()

        # IMPORTANT:
        # Buffers move with model.to(device) but are NOT parameters.
        # persistent=False reduces checkpoint size and is appropriate for RoPE.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _cache_rope_freq(self):
        """
        Precompute frequencies for RoPE.
        No angle recomputation during forward().
        """

        # position index: (S,)
        position_ids = torch.arange(0, self.max_seq_len, dtype=torch.float32)

        # number of 2D rotation blocks
        num_slices = self.d_k // 2

        # slice index: (D/2,)
        slice_idx = torch.arange(0, num_slices, dtype=torch.float32)

        # inverse frequency for each 2D block: (D/2,)
        inv_freq = 1.0 / (self.theta ** (slice_idx / num_slices))

        # reshape for broadcast: (S,1) and (1,D/2)
        position_ids = einops.rearrange(position_ids, "p -> p 1")
        inv_freq     = einops.rearrange(inv_freq,     "j -> 1 j")

        # theta[p,j] = p * inv_freq[j]   → shape (S, D/2)
        theta = position_ids * inv_freq

        cos = theta.cos()
        sin = theta.sin()

        # final shape expected by forward():
        #   (1, S, 1, D/2) to broadcast over (B,S,H,D/2)
        # cos = einops.rearrange(cos, "p j -> 1 p 1 j")
        # sin = einops.rearrange(sin, "p j -> 1 p 1 j")

        return cos, sin

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        x: (..., seq_len, d_k)
        Returns rotated x: (..., seq_len)
        """
        assert x.device == self.cos.device, "device mismatch"
    
        # shapes: (B,S,H,D/2)
        x_even = x[..., 0::2] # starting from 0 and take 2 steps
        x_odd  = x[..., 1::2] # starting from 1 and take 2 steps

        cos, sin = self.cos[token_positions], self.sin[token_positions]

        # 2D rotation:
        # [xe]   [ cos  -sin ] [xe]
        # [xo] = [ sin   cos ] [xo]
        x_even_rot = cos * x_even - sin * x_odd
        x_odd_rot  = cos * x_odd  + sin * x_even

        # Reconstruct interleaved (B,S,H,D)
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_even_rot
        x_rot[..., 1::2]  = x_odd_rot

        return x_rot.to(x.dtype)
