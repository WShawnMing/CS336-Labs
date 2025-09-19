import torch
from cs336_basics.module.attention import MultiHeadAttentionWithRoPE
from cs336_basics.module.linear import Linear
from cs336_basics.module.swiglu import SwiGLU
from cs336_basics.module.rope import RoPE
from cs336_basics.module.softmax import Softmax
from cs336_basics.module.rms_norm import RMSNorm

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, max_seq_len: int, theta: float, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        

        self.attn = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, self.device, self.dtype)
        self.ffn = SwiGLU(d_model, d_ff, self.device, self.dtype)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
        o1 = self.ln1(in_features)
        o2 = self.attn(o1, token_positions)
        o3 = in_features + o2

        o4 = self.ln2(o3)
        o5 = self.ffn(o4)
        o6 = o3 + o5

        return o6