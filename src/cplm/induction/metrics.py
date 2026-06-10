from __future__ import annotations

from typing import Dict, List
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import InductionVocab, make_batch


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


@torch.no_grad()
def evaluate_recall(
    model: nn.Module,
    *,
    vocab: InductionVocab,
    n_pairs: int,
    n_queries: int = 1,
    batch_size: int = 256,
    n_batches: int = 4,
    device: str | torch.device,
    role_mode: str = "shared",
    sequence_mode: str = "query",
) -> Dict[str, float]:
    model.eval()
    if sequence_mode == "copy_repeat":
        n_queries = max(1, n_pairs - 1)
    correct = 0
    total = 0
    loss_sum = 0.0
    margin_sum = 0.0
    induction_mass_sum = 0.0
    induction_head_max_sum = 0.0
    by_dist: Dict[int, List[int]] = {d: [0, 0] for d in range(n_pairs)}

    q_indices = list(range(n_pairs))
    for _ in range(n_batches):
        batch = make_batch(
            batch_size=batch_size,
            n_pairs=n_pairs,
            n_queries=n_queries,
            vocab=vocab,
            device=device,
            query_pair_indices=q_indices,
            role_mode=role_mode,
            sequence_mode=sequence_mode,
        )
        out = model(batch["input_ids"], return_attn=True)
        logits = out["logits"]
        bsz = logits.size(0)
        b_idx = torch.arange(bsz, device=logits.device)[:, None].expand(bsz, n_queries)
        q_idx = torch.arange(n_queries, device=logits.device)[None, :].expand(bsz, n_queries)
        qpos = batch["query_pos"]
        q_logits = logits[b_idx, qpos]  # [B,Q,V]
        target = batch["target_value"]
        flat_logits = q_logits.reshape(-1, q_logits.size(-1))
        flat_target = target.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_target, reduction="sum")
        preds = q_logits.argmax(dim=-1)
        ok = (preds == target)
        correct += int(ok.sum().item())
        total += int(target.numel())
        loss_sum += float(loss.item())

        flat_indices = torch.arange(flat_target.numel(), device=logits.device)
        target_logit = flat_logits[flat_indices, flat_target]
        masked = flat_logits.clone()
        masked[flat_indices, flat_target] = -float("inf")
        margin = target_logit - masked.max(dim=-1).values
        margin_sum += float(margin.sum().item())

        if out.get("attentions"):
            value_pos = batch["value_pos"]
            masses = []
            max_head_masses = []
            for attn in out["attentions"]:
                # Gather [B,Q,H] attention from each query key to its previous value.
                head_ids = torch.arange(attn.size(1), device=attn.device)[None, None, :].expand(bsz, n_queries, attn.size(1))
                bb = b_idx[:, :, None].expand_as(head_ids)
                qqpos = qpos[:, :, None].expand_as(head_ids)
                vvpos = value_pos[:, :, None].expand_as(head_ids)
                layer_mass = attn[bb, head_ids, qqpos, vvpos]  # [B,Q,H]
                masses.append(layer_mass.mean(dim=2))
                max_head_masses.append(layer_mass.max(dim=2).values)
            mass = torch.stack(masses, dim=0).mean(dim=0)  # [B,Q]
            max_head_mass = torch.stack(max_head_masses, dim=0).max(dim=0).values
            induction_mass_sum += float(mass.sum().item())
            induction_head_max_sum += float(max_head_mass.sum().item())

        for d in range(n_pairs):
            mask = batch["pair_distance"] == d
            if int(mask.sum().item()) > 0:
                by_dist[d][0] += int(ok[mask].sum().item())
                by_dist[d][1] += int(mask.sum().item())

    avg_loss = loss_sum / max(total, 1)
    out = {
        "recall_accuracy": correct / max(total, 1),
        "recall_loss": avg_loss,
        "recall_margin": margin_sum / max(total, 1),
        "induction_score": induction_mass_sum / max(total, 1),
        "induction_head_max_score": induction_head_max_sum / max(total, 1),
        "loss_reduction_vs_uniform": math.log(vocab.n_symbols) - avg_loss,
        "floor_accuracy": 1.0 / float(vocab.n_symbols),
        "value_role_floor_accuracy": (2.0 / float(vocab.n_symbols)) if role_mode == "split" else (1.0 / float(vocab.n_symbols)),
    }
    for d, (c, n) in by_dist.items():
        out[f"acc_dist_{d}"] = c / n if n else float("nan")
    return out
