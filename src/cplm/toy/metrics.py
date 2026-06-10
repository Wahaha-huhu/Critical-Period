from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List
import math
import numpy as np
import torch
from torch import nn

from .data import BASE_LABEL_IDS, CLASS_LABEL_IDS, ToyBatch, full_factorial_eval_batch


@torch.no_grad()
def evaluate_task(model: nn.Module, task: str, device: str | torch.device = "cpu") -> Dict[str, float]:
    model.eval()
    batch = full_factorial_eval_batch(task, device=device, repeats=256)
    logits = model(batch.input_ids)
    pred_pos = batch.label_pos - 1  # logits after Q token predict label token.
    if task == "base":
        label_ids = BASE_LABEL_IDS.to(device)
        correct_class = batch.type_ids
    else:
        label_ids = CLASS_LABEL_IDS.to(device)
        correct_class = batch.class_ids
    pair_logits = logits[:, pred_pos, label_ids]
    pred = pair_logits.argmax(dim=-1)
    acc = (pred == correct_class).float().mean().item()
    margin = (pair_logits.gather(1, correct_class[:, None]).squeeze(1) - pair_logits.gather(1, (1 - correct_class)[:, None]).squeeze(1)).mean().item()
    loss = model.loss(batch.input_ids, batch.labels).item()
    return {"accuracy": acc, "margin": margin, "loss": loss}


def update_norms(before: Dict[str, torch.Tensor], after_model: nn.Module) -> Dict[str, float]:
    total_sq = 0.0
    emb_sq = 0.0
    deep_sq = 0.0
    for name, p in after_model.named_parameters():
        if name not in before:
            continue
        d = (p.detach().cpu() - before[name]).float()
        val = float((d * d).sum().item())
        total_sq += val
        if "embed" in name or "lm_head" in name:
            emb_sq += val
        else:
            deep_sq += val
    total = math.sqrt(total_sq)
    return {
        "update_norm_total": total,
        "update_norm_embedding_head": math.sqrt(emb_sq),
        "update_norm_deep": math.sqrt(deep_sq),
        "update_fraction_deep": (math.sqrt(deep_sq) / total) if total > 0 else float("nan"),
    }
