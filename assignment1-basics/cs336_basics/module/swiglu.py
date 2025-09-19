import torch
from cs336_basics.module.linear import Linear
class SiLU(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = 1/(1+torch.exp(-x))
        return x * s

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.silu = SiLU(device=device, dtype=dtype)
        # (in,out)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff,d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff,  device=device, dtype=dtype) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
