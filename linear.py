import torch

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
