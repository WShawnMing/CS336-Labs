import torch
from collections.abc import Iterable


# def get_grad_norm(parameters: torch.nn.Parameter) -> float:
#     """
#     Computes the norm of the gradients of an iterable of parameters.
#     """
#     return torch.norm(parameters.grad.data, 2)

def clip_grad(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    """
    norm = 0
    for p in parameters:
        if p.grad is None:
            continue    
        norm += torch.sum(p.grad.data ** 2)

    norm = torch.sqrt(norm)

    if norm >= max_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data * (max_norm / norm)

    return max_norm
