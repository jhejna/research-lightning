"""
This file contains schedule functions that can be used as learning rate schedules

All learning rate schedulers use the pytorch LambdaLR function and any additional kwargs.
"""
import math


def linear_decay(total_steps: int, start_step: int = 1, offset: int = 0):
    def fn(step):
        return 1.0 - max(0, step + offset - start_step) / (total_steps - start_step)

    return fn


def linear_warmup(total_steps: int, multiplier: float = 1.0):
    def fn(step):
        return multiplier * min(1.0, step / total_steps)

    return fn


def cosine_with_linear_warmup(warmup_steps: int, total_steps: int, num_cycles: float = 0.5, min_lr_ratio=1e-1):
    def fn(step):
        step = min(step, total_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        out = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        if out < min_lr_ratio:
            return min_lr_ratio
        else:
            return out

    return fn
