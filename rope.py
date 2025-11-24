import torch
import einops

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Implements RoPE (Su et al., 2021).
    Rotates Q/K vectors in 2D slices of size 2.

    Shapes:
      - cos, sin: (1, max_seq_len, 1, d_k/2)
      - x (input): (B, S, H, D)
      - output:    (B, S, H, D)
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
        cos = einops.rearrange(cos, "p j -> 1 p 1 j")
        sin = einops.rearrange(sin, "p j -> 1 p 1 j")

        return cos, sin

    def forward(self, x: torch.Tensor):
        """
        x: (B, S, H, D)
        Returns rotated x: (B, S, H, D)
        """

        B, S, H, D = x.shape
        assert S == self.max_seq_len, "sequence length mismatch"
        assert D == self.d_k, "head dim mismatch"
        assert x.device == self.cos.device, "device mismatch"

        # Select even/odd indices for 2D rotation
        idx = torch.arange(0, D, device=x.device)
        even_mask = (idx % 2) == 0
        odd_mask  = ~even_mask

        # shapes: (B,S,H,D/2)
        x_even = x[..., even_mask]
        x_odd  = x[..., odd_mask]

        # 2D rotation:
        # [xe]   [ cos  -sin ] [xe]
        # [xo] = [ sin   cos ] [xo]
        x_even_rot = self.cos * x_even - self.sin * x_odd
        x_odd_rot  = self.cos * x_odd  + self.sin * x_even

        # Reconstruct interleaved (B,S,H,D)
        x_rot = torch.empty_like(x)
        x_rot[..., even_mask] = x_even_rot
        x_rot[..., odd_mask]  = x_odd_rot

        return x_rot.to(x.dtype)
