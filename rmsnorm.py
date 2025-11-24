import torch
import einops

class RMSNorm(torch.nn.Module):
    """
    RMSNorm: x / RMS(x) * g
    RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(d_model))  # IMPORTANT: initialize to 1

    def forward(self, x):
        assert x.shape[-1] == self.d_model, "shape mismatch"

        dtype_in = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            einops.reduce(x * x, "... d -> ... 1", reduction="mean") + self.eps
        )

        out = (x / rms) * self.g  # broadcasts (..., d_model)
        return out.to(dtype_in)
