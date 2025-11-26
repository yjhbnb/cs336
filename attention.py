import torch
import einops


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


# quick sanity check like yours
if __name__ == "__main__":
    Q, K = torch.rand((10, 8, 20, 64)), torch.rand((10, 8, 20, 64))
    V = torch.rand((10, 8, 20, 32))
    mask = torch.triu(torch.ones(20, 20, dtype=torch.bool))

    print(scaled_dot_product_attention(Q, K, V, mask).shape)
    # expected: torch.Size([10, 8, 20, 32])
