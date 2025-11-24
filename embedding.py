import torch

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
