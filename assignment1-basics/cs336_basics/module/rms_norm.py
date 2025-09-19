import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        # x [b,s,h]
        # weight [h]
        X = x.to(torch.float32)
        x2 = X * X
        X = X / torch.sqrt(x2.mean(dim=-1, keepdim=True) + self.eps)
        X = X * self.weight.to(X.dtype)
        return X.to(x.dtype)