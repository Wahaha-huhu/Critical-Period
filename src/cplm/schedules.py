from __future__ import annotations

import math


def lr_at_step(
    step: int,
    schedule: str,
    total_steps: int,
    peak_lr: float,
    warmup_fraction: float,
    terminal_lr_factor: float,
) -> float:
    """Learning-rate schedule used for optimizer updates.

    Step is one-indexed for update semantics: call with current training step + 1.
    For S1 beyond total_steps, hold the terminal learning rate. This implements
    the fixed-dose continuation rule.
    """
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    terminal_lr = peak_lr * terminal_lr_factor
    if step <= warmup_steps:
        return peak_lr * step / warmup_steps
    if schedule == "s2_constant":
        return peak_lr
    if schedule != "s1_decay":
        raise ValueError(f"Unknown schedule: {schedule}")
    if step >= total_steps:
        return terminal_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return terminal_lr + (peak_lr - terminal_lr) * cosine


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
