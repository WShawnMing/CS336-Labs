
import torch
from einops import rearrange, einsum, reduce
import math
from cs336_basics.module.softmax import Softmax
from cs336_basics.module.linear import Linear
from cs336_basics.module.rope import RoPE

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=-1)  
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = torch.einsum("... i d, ... j d -> ... i j", Q, K) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=Q.dtype, device=Q.device))
        if mask is not None:
            scores = scores.masked_fill(mask==True, float("-inf"))
        attn_weights = self.softmax(scores)
        return torch.einsum("... i j, ... j d -> ... i d", attn_weights, V)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.ScaledDotProductAttention = ScaledDotProductAttention()

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        s= in_features.shape[-2]
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)

        Q = Q.view(-1,s,self.num_heads,self.d_model//self.num_heads).transpose(1, 2)
        K = K.view(-1,s,self.num_heads,self.d_model//self.num_heads).transpose(1, 2)
        V = V.view(-1,s,self.num_heads,self.d_model//self.num_heads).transpose(1, 2)


        causal_mask = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=Q.device), diagonal=1
        )

        attn_out = self.ScaledDotProductAttention(Q, K, V, causal_mask)
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")
        out = self.o_proj(attn_out)
        return out


class MultiHeadAttentionWithRoPE(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,max_seq_len: int, theta: float, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.ScaledDotProductAttention = ScaledDotProductAttention()

        self.max_seq_len = max_seq_len
        self.theta = theta
        # 注意这里d_model//num_heads 是每个head的维度
        self.RoPE = RoPE(theta, d_model//num_heads, max_seq_len)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        b,s,_ = in_features.shape
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        # 先reshape 再RoPE
        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)

        causal_mask = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=Q.device), diagonal=1
        )

        attn_out = self.ScaledDotProductAttention(Q, K, V, causal_mask)
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")
        out = self.output_proj(attn_out)
        return out
        