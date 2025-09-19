import torch

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 创建位置索引
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device).unsqueeze(1) # "max_seq_len,1"
        dim_indices = torch.arange(0, self.d_k, 2, dtype=torch.float32, device=self.device) # "d_k//2"
        freqs= self.theta ** (-dim_indices / self.d_k) # "d_k//2"

        angles = positions * freqs

        # 计算余弦和正弦值  (max_seq_len, d_k//2)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        # "max_seq_len,dk"
        self.cos_val = torch.zeros(max_seq_len, d_k, dtype=torch.float32, device=device)
        self.sin_val = torch.zeros(max_seq_len, d_k, dtype=torch.float32, device=device)

        self.cos_val[:,0::2] = cos_vals
        self.cos_val[:,1::2] = cos_vals

        self.sin_val[:,0::2] = -sin_vals
        self.sin_val[:,1::2] = sin_vals

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # " ... sequence_length d_k"
        # " ... sequence_length"
        # res = x * self.cos_val + x_reverse * self.sin_val

        assert token_positions.shape[-1] == x.shape[-2]
        cos_vals = self.cos_val[token_positions]
        sin_vals = self.sin_val[token_positions]

        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x[..., 1::2]  # 偶数位置取奇数维度的负值
        x_rotated[..., 1::2] = x[..., 0::2]   # 奇数位置取偶数维度的值

        return x * cos_vals + x_rotated * sin_vals