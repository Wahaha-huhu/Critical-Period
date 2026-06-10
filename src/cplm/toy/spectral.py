from __future__ import annotations

from typing import Dict, List
import math
import numpy as np
import torch
from torch import nn


def _effective_rank(s: torch.Tensor) -> float:
    s = s.detach().float().cpu()
    if s.numel() == 0 or float(s.sum()) <= 0:
        return float("nan")
    p = s / s.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum().item()
    return float(math.exp(entropy))


def matrix_spectral_metrics(w: torch.Tensor) -> Dict[str, float]:
    w = w.detach().float().cpu()
    if w.ndim != 2:
        return {}
    try:
        s = torch.linalg.svdvals(w)
    except RuntimeError:
        return {}
    fro_sq = float((s * s).sum().item())
    top = float(s.max().item()) if s.numel() else float("nan")
    stable_rank = fro_sq / (top * top + 1e-12) if top == top else float("nan")
    return {
        "spectral_norm": top,
        "stable_rank": stable_rank,
        "effective_rank": _effective_rank(s),
    }


def model_spectral_summary(model: nn.Module) -> Dict[str, float]:
    vals: Dict[str, List[float]] = {"spectral_norm": [], "stable_rank": [], "effective_rank": []}
    for name, p in model.named_parameters():
        if p.ndim != 2 or any(skip in name for skip in ["token_embed", "pos_embed"]):
            continue
        m = matrix_spectral_metrics(p)
        for k, v in m.items():
            if v == v:
                vals[k].append(v)
    out: Dict[str, float] = {}
    for k, arr in vals.items():
        if arr:
            out[f"mean_{k}"] = float(np.mean(arr))
            out[f"min_{k}"] = float(np.min(arr))
            out[f"max_{k}"] = float(np.max(arr))
        else:
            out[f"mean_{k}"] = float("nan")
    return out
