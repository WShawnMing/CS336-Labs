import torch
import math

class LRCosineSchedule:
    def __init__(self, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it):
        if it < self.warmup_iters:
            return self.max_learning_rate * it / self.warmup_iters

        elif self.warmup_iters <= it <= self.cosine_cycle_iters:
            return self.min_learning_rate + 1/2 * (1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters)))* (self.max_learning_rate - self.min_learning_rate)

        else:
            return self.min_learning_rate
