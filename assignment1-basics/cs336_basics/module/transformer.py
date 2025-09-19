import torch
from cs336_basics.module.transformer_block import TransformerBlock
from cs336_basics.module.embedding import Embedding
from cs336_basics.module.linear import Linear
from cs336_basics.module.rms_norm import RMSNorm

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.max_seq_len = context_length
        self.theta = rope_theta
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, d_ff, num_heads, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits