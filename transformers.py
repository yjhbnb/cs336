import torch
from building_blocks import *

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)  # (B, S) -> (B, S, d_model)
        for layer in self.layers:
            x = layer(x)  # (B, S, d_model)
        x = self.ln_final(x)  # (B, S, d_model)
        logits = self.lm_head(x)  # (B, S, vocab_size)
        return logits