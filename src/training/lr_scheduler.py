"""
Learning Rate Schedulers.

Warmup + cosine annealing, linear warmup + decay.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def create_lr_scheduler(optimizer: Optimizer, scheduler_type: str = "cosine_with_warmup",
                        warmup_steps: int = 1000, total_steps: int = 100000,
                        min_lr: float = 1e-7) -> _LRScheduler:
    """Factory for LR schedulers."""
    if scheduler_type == "cosine_with_warmup":
        return CosineWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr)
    elif scheduler_type == "linear_with_warmup":
        return LinearWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr)
    elif scheduler_type == "constant_with_warmup":
        return ConstantWarmupScheduler(optimizer, warmup_steps)
    raise ValueError(f"Unknown scheduler: {scheduler_type}")


class CosineWarmupScheduler(LambdaLR):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 total_steps: int, min_lr: float = 1e-7):
        base_lr = optimizer.defaults["lr"]
        min_ratio = min_lr / base_lr if base_lr > 0 else 0

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        super().__init__(optimizer, lr_lambda)


class LinearWarmupScheduler(LambdaLR):
    """Linear warmup followed by linear decay."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int,
                 total_steps: int, min_lr: float = 1e-7):
        base_lr = optimizer.defaults["lr"]
        min_ratio = min_lr / base_lr if base_lr > 0 else 0

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_ratio, 1.0 - progress)

        super().__init__(optimizer, lr_lambda)


class ConstantWarmupScheduler(LambdaLR):
    """Linear warmup followed by constant LR."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        super().__init__(optimizer, lr_lambda)
