import torch    

class Softmax(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(x, dim=self.dim, keepdim=True)[0]
        exp = torch.exp(x - max_val)
        s = torch.sum(exp, dim=self.dim, keepdim=True)
        return exp / s