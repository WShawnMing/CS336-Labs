from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                t = state.get("t", 0)
                if p.grad is None:
                    continue


                g = p.grad.data

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                m = group["beta1"] * m + (1 - group["beta1"]) * g
                v = group["beta2"] * v + (1 - group["beta2"]) * g**2

                lr = group["lr"] * math.sqrt(1 - group["beta2"]**(t + 1)) / (1 - group["beta1"]**(t + 1))

                p.data -= lr * m / (torch.sqrt(v) + group["eps"]) 
                p.data -= group["lr"] * group["weight_decay"] * p.data


                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
    