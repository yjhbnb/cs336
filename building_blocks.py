import torch
import einops

class Linear(torch.nn.Module):
    """
    Custom Linear layer.
    Matches PyTorch nn.Linear convention:
      weight shape = (out_features, in_features)
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = self._set_linear_weight()   # (out, in)

        # optional bias: (out,)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_features))
        else:
            self.bias = None

    def _set_linear_weight(self):
        """
        Weight init: truncated normal.
        Shape: (out, in)
        """
        weight = torch.nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )

        # conservative init similar to Glorot
        std = 2.0 / (self.in_features + self.out_features)
        torch.nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )

        return weight

    def forward(self, x):
        """
        x: (..., in_features)
        returns: (..., out_features)
        """

        assert x.shape[-1] == self.in_features, "shape mismatch"
        assert x.device == self.weight.device, "device mismatch"

        # Matmul with transposed weight: (in) @ (in→out)
        out = x @ self.weight.T

        if self.bias is not None:
            out = out + self.bias   # bias broadcasts

        return out.to(x.dtype)


class Embedding(torch.nn.Module):
    """
    Custom embedding layer.
    Equivalent to nn.Embedding without padding_idx.

    Weight shape: (num_embeddings, embedding_dim)
      where row i is the vector for token i.

    Forward:
      x: (B, S) integer token ids
      output: (B, S, embedding_dim)
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = self._init_weight()

    def _init_weight(self):
        """
        Simple truncated normal init.
        """
        weight = torch.nn.Parameter(
            torch.empty(self.num_embeddings, self.embedding_dim)
        )

        torch.nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

        return weight

    def forward(self, x):
        """
        x: (B, S) integer token ids
        returns: (B, S, embedding_dim)

        Uses PyTorch's advanced indexing:
        weight[x] gathers rows; no one-hot involved.
        """
        return self.weight[x]

class RMSNorm(torch.nn.Module):
    """
    RMSNorm: x / RMS(x) * weight
    RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))  # IMPORTANT: initialize to 1

    def forward(self, x):
        assert x.shape[-1] == self.d_model, "shape mismatch"

        dtype_in = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            einops.reduce(x * x, "... d -> ... 1", reduction="mean") + self.eps
        )

        out = (x / rms) * self.weight  # broadcasts (..., d_model)
        return out.to(dtype_in)

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
        token_positions: (..., seq_len)
        Returns rotated x: (..., seq_len)
        """
        assert x.device == self.cos.device, "device mismatch"
    
        # shapes: (..., seq_len, D/2)
        x_even = x[..., 0::2] # starting from 0 and take 2 steps
        x_odd  = x[..., 1::2] # starting from 1 and take 2 steps

        # cos in shape (..., seq_len, D/2), token_positions: (..., seq_len)
        # broadcasted shape is (..., seq_len, D/2)
        cos, sin = self.cos[token_positions], self.sin[token_positions]

        # 2D rotation:
        # [xe]   [ cos  -sin ] [xe]
        # [xo] = [ sin   cos ] [xo]
        x_even_rot = cos * x_even - sin * x_odd # shape (..., seq_len, D/2)
        x_odd_rot  = cos * x_odd  + sin * x_even # shape (..., seq_len, D/2)

        # Reconstruct interleaved (..., seq_len, D)
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_even_rot
        x_rot[..., 1::2]  = x_odd_rot

        return x_rot.to(x.dtype)


class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # self.d_ff = self.d_model * 8 // 3

        self.w1   = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, bias=False)
        self.w2  = Linear(in_features=self.d_ff, out_features=self.d_model, bias=False)

    def forward(self, x):
        assert x.shape[-1] == self.d_model, "shape mismatched"

        silu_ffn_x = self.w1(x)                # (..., d_ff)
        silu_x     = silu_ffn_x * torch.sigmoid(silu_ffn_x)  # SiLU(...), (..., d_ff)

        normal_x   = silu_x * self.w3(x)     # gate ⊙ W3 x, (..., d_ff)
        out        = self.w2(normal_x)        # (..., d_model)

        return out.to(x.dtype)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # subtract max along the given dim for numerical stability
    # (exp of large positive numbers would otherwise overflow)
    offset = torch.max(x, dim=dim, keepdim=True)[0]  # [0] are values, [1] would be indices
    offset_x = x - offset

    # standard softmax: exp(shifted) / sum(exp(shifted)) along dim
    exp_offset_x = offset_x.exp()
    out = exp_offset_x / torch.sum(exp_offset_x, dim=dim, keepdim=True)

    # keep the same dtype as the input (important if e.g. using float16/bfloat16)
    return out.to(x.dtype)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    offset = torch.max(logits, dim=-1, keepdim=True)[0] # 0 for values and 1 for indices
    shifted_logits = logits - offset
    batch_size = logits.shape[0]
    target_logits = shifted_logits[torch.arange(batch_size), targets]
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    loss = -target_logits + log_sum_exp
    return loss.mean()


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Q and K must have the same shape: (batch, ..., seq_len, d_k)
    assert Q.shape == K.shape, "shape mismatched between Q and K"

    # d_k is the per-head key/query dimension (last dimension)
    d_k = Q.shape[-1]

    # move seq_len to the second-to-last position and d_k to the last
    # for K we also swap seq_len and d_k so that we can do Q @ K_swapped:
    #   Q:       (batch ... seq dim)
    #   K_swapped: (batch ... dim seq)
    # result:   (batch ... seq seq)   <-- attention scores between tokens
    K_swapped = einops.rearrange(K, "batch ... seq dim -> batch ... dim seq")
    scores = Q @ K_swapped  # (batch ... seq seq)

    if mask is not None:
        # mask has shape (seq, seq). By convention here:
        #   mask[i, j] == True  → position j is allowed to be attended to by query i
        #   mask[i, j] == False → position j should get attention probability 0
        #
        # We apply the mask BEFORE softmax by turning disallowed positions into -inf.
        # Because exp(-inf) = 0, those entries will get probability 0 after softmax.
        # masked_fill(...) fills entries where the mask is True, so we negate it (~mask)
        scores = scores.masked_fill(mask=~mask, value=float("-inf"))

    # scale by sqrt(d_k) as in "Attention is All You Need" to keep logits well-scaled
    # softmax is applied over the last dimension (the "key" dimension) so that
    # for each query token, probabilities over all keys sum to 1.
    scores = softmax(scores / (d_k ** 0.5), dim=-1)  # (batch ... seq seq)

    # finally, weight values V by the attention probabilities:
    #   scores: (batch ... seq seq)
    #   V:      (batch ... seq d_v)
    # result:   (batch ... seq d_v)  <-- same shape convention as Q/K but with d_v
    out = scores @ V  # (batch ... seq d_v)

    # match Q's dtype (again useful for mixed precision)
    return out.to(Q.dtype)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        # project into Q, K, V for all heads at once: (B, S, d_model) -> (B, S, h * d_k)
        self.q_proj = Linear(in_features=self.d_model, out_features=self.num_heads * self.d_k)
        self.k_proj = Linear(in_features=self.d_model, out_features=self.num_heads * self.d_k)
        self.v_proj = Linear(in_features=self.d_model, out_features=self.num_heads * self.d_v)

        # output projection: (B, S, h * d_v) -> (B, S, d_model)
        self.output_proj = Linear(in_features=self.num_heads * self.d_v, out_features=self.d_model)

        self.theta = theta
        self.max_seq_len = max_seq_len
        self.rope = RotaryPositionalEmbedding(theta=self.theta,
                                              d_k=self.d_k,
                                              max_seq_len=self.max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, S, d_model)
        token_positions (optional): (..., S) absolute positions. If None, use 0..S-1.
        """
        B, S, D = x.shape
        assert D == self.d_model, "shape mismatched"

        # positions: [0, 1, ..., S-1] for each sequence in the batch
        if token_positions is None:
            token_positions = einops.rearrange(torch.arange(S, device=x.device), "s -> 1 s")

        # 1) Linear projections
        Q = self.q_proj(x)  # (B, S, h * d_k)
        K = self.k_proj(x)  # (B, S, h * d_k)
        V = self.v_proj(x)  # (B, S, h * d_v)

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
        out = self.output_proj(att)  # (B, S, d_model)

        return out.to(x.dtype)


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer (single-parameter-group version).

    Notation vs paper / handout:
      - theta (θ)      -> parameter tensor `p`
      - alpha (α)      -> base learning rate `alpha`
      - alpha_t (α_t)  -> bias-corrected learning rate `alpha_t`
      - beta1 (β₁)     -> `beta1`
      - beta2 (β₂)     -> `beta2`
      - m_t            -> first moment (exp. moving avg of g), `m`
      - v_t            -> second moment (exp. moving avg of g²), `v`
      - lambda (λ)     -> weight decay `lbd`
      - eps (ε)        -> numerical stability term `eps`
      - t              -> step counter per parameter, `state["t"]`
    """

    def __init__(
        self,
        params,
        alpha: float = 1e-3,   # α_max in the handout (base LR)
        beta1: float = 0.9,    # β₁
        beta2: float = 0.95,   # β₂
        eps: float = 1e-8,     # ε
        lbd: float = 0.01,     # λ (weight decay)
        max_norm: float | None = None,  # global grad clip ‖g‖₂ ≤ max_norm
    ):
        # Hyperparameters stored in param_groups (not in state)
        defaults = {
            "alpha": alpha,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "lambda": lbd,
        }
        super().__init__(params, defaults)
        self.max_norm = max_norm

    def step(self, closure=None):
        # Optional closure to recompute loss if needed (standard Optimizer API)
        loss = None if closure is None else closure()

        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lbd = group["lambda"]

            # 1) Global gradient clipping (if enabled): scales all p.grad
            if self.max_norm is not None:
                gradient_clipping(group["params"], max_norm=self.max_norm)

            # 2) AdamW update per parameter
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["t"] = 1                        # t = 1 (as in handout)
                    state["m"] = torch.zeros_like(p)      # m₀ = 0
                    state["v"] = torch.zeros_like(p)      # v₀ = 0

                t = state["t"]       # current step t
                m = state["m"]       # m_{t-1}
                v = state["v"]       # v_{t-1}

                grad = p.grad        # g_t = ∇_θ ℓ(θ; B_t)

                # m_t = β₁ m_{t-1} + (1 - β₁) g_t
                m = beta1 * m + (1.0 - beta1) * grad

                # v_t = β₂ v_{t-1} + (1 - β₂) g_t²
                v = beta2 * v + (1.0 - beta2) * grad.square()

                # Bias-corrected learning rate:
                # α_t = α * sqrt(1 - β₂^t) / (1 - β₁^t)
                alpha_t = alpha * ((1.0 - beta2**t) ** 0.5) / (1.0 - beta1**t)

                # θ_t = θ_{t-1} - α_t * m_t / sqrt(v_t + ε)
                # θ_t = θ_t      - α * λ * θ_t      (decoupled weight decay)
                with torch.no_grad():
                    p -= alpha_t * m / (v + eps).sqrt()
                    p -= alpha * lbd * p

                # Increment step and store updated moments
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def learning_rate_schedule(t, alpha_min, alpha_max, T_w, T_c):
    import numpy as np
    if t < T_w:
        alpha_t = t * alpha_max / T_w
    elif t >= T_w and t <= T_c:
        alpha_t = alpha_min + 0.5 * (1 + np.cos((t - T_w) * np.pi / (T_c - T_w))) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t


def gradient_clipping(params, max_norm, eps=1e-6):
    # 1. Compute global L2 norm
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += p.grad.pow(2).sum()
    total_norm = total_norm.sqrt()

    # 2. Compute scale factor
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)

        # 3. Scale gradients in place
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)   # in-place gradient modification
