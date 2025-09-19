import torch
import math

# No bias Linear
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std = math.sqrt(2 / (in_features + out_features))
        weight_tensor = torch.nn.init.trunc_normal_(torch.empty(out_features, in_features, device=device, dtype=dtype), mean=0, std=std)
        self.weight = torch.nn.Parameter(weight_tensor)
        # self.bias = torch.nn.Parameter(torch.randn(out_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor  :
        # x: (..., in_features)
        # weight: (out_features, in_features)
        return  torch.einsum("...i,oi->...o", x, self.weight)