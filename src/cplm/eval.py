from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F


def evaluate_probe(
    model: torch.nn.Module,
    probe_items: list[dict[str, Any]],
    pad_id: int,
    device: torch.device | str,
    batch_size: int = 256,
) -> dict[str, float]:
    model.eval()
    template_correct: dict[str, list[bool]] = defaultdict(list)
    template_margin: dict[str, list[float]] = defaultdict(list)
    by_item: dict[int, dict[str, bool]] = defaultdict(dict)

    with torch.no_grad():
        for start in range(0, len(probe_items), batch_size):
            chunk = probe_items[start : start + batch_size]
            max_len = max(len(item["tokens"]) - 1 for item in chunk)
            input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
            pred_pos = torch.empty(len(chunk), dtype=torch.long, device=device)
            correct_ids = torch.empty(len(chunk), dtype=torch.long, device=device)
            incorrect_ids = torch.empty(len(chunk), dtype=torch.long, device=device)
            for i, item in enumerate(chunk):
                ids = item["tokens"][:-1]
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                pred_pos[i] = int(item["verb_pos"]) - 1
                correct_ids[i] = int(item["correct_id"])
                incorrect_ids[i] = int(item["incorrect_id"])
            logits = model(input_ids)["logits"]
            rows = torch.arange(len(chunk), device=device)
            selected = logits[rows, pred_pos]
            logp = F.log_softmax(selected, dim=-1)
            margins = (logp[rows, correct_ids] - logp[rows, incorrect_ids]).detach().cpu().tolist()
            for item, margin in zip(chunk, margins, strict=True):
                correct = float(margin) > 0.0
                template = item["template"]
                template_correct[template].append(correct)
                template_margin[template].append(float(margin))
                by_item[int(item["item_id"])][template] = correct

    metrics: dict[str, float] = {}
    for template in ("local", "pp_same", "pp_opp"):
        vals = template_correct.get(template, [])
        margins = template_margin.get(template, [])
        metrics[f"acc_{template}"] = float(sum(vals) / len(vals)) if vals else float("nan")
        metrics[f"margin_{template}"] = float(sum(margins) / len(margins)) if margins else float("nan")
    inv_vals = []
    for item_templates in by_item.values():
        if "pp_same" in item_templates and "pp_opp" in item_templates:
            inv_vals.append(item_templates["pp_same"] and item_templates["pp_opp"])
    metrics["invariance"] = float(sum(inv_vals) / len(inv_vals)) if inv_vals else float("nan")
    model.train()
    return metrics
