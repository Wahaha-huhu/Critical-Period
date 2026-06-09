from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from cplm.models.transformer import DecoderOnlyTransformer

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


@dataclass
class TextVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls, texts: Iterable[str]) -> "TextVocab":
        specials = ["<pad>", "<bos>", "<eos>"]
        seen = list(specials)
        have = set(seen)
        for text in texts:
            for tok in tokenize_text(text):
                if tok not in have:
                    have.add(tok)
                    seen.append(tok)
        return cls(token_to_id={t: i for i, t in enumerate(seen)}, id_to_token=seen)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.token_to_id[t] for t in tokenize_text(text)]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def encode_tokens(self, tokens: list[str], add_special: bool = True) -> list[int]:
        ids = [self.token_to_id[t] for t in tokens]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def to_jsonable(self) -> dict[str, Any]:
        return {"n_tokens": len(self.id_to_token), "tokens": self.id_to_token}


class PackedTextDataset:
    def __init__(self, sequences: list[list[int]], pad_id: int, context_length: int, seed: int = 0) -> None:
        self.sequences = sequences
        self.pad_id = int(pad_id)
        self.context_length = int(context_length)
        self.rng = random.Random(seed)
        if not sequences:
            raise ValueError("PackedTextDataset requires at least one sequence")

    def make_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = torch.full((batch_size, self.context_length), self.pad_id, dtype=torch.long)
        y = torch.full((batch_size, self.context_length), -100, dtype=torch.long)
        token_count = 0
        for b in range(batch_size):
            seq = self.rng.choice(self.sequences)
            if len(seq) > self.context_length:
                start = self.rng.randint(0, len(seq) - self.context_length)
                seq = seq[start : start + self.context_length]
            L = min(len(seq), self.context_length)
            x[b, :L] = torch.tensor(seq[:L], dtype=torch.long)
            if L > 1:
                y[b, : L - 1] = x[b, 1:L]
                token_count += L - 1
        return x.to(device), y.to(device), token_count


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def linear_warmup_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(base_lr)
    return float(base_lr) * min(1.0, step / warmup_steps)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable, falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def maybe_autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext
    return nullcontext()


def build_tiny_model(config: dict[str, Any], vocab_size: int) -> DecoderOnlyTransformer:
    m = config["model"]
    return DecoderOnlyTransformer(
        vocab_size=vocab_size,
        n_layers=int(m["n_layers"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        context_length=int(m["context_length"]),
        mlp_ratio=int(m.get("mlp_ratio", 4)),
        dropout=float(m.get("dropout", 0.0)),
    )


@torch.no_grad()
def next_token_logprobs(model: torch.nn.Module, prefix_ids: list[int], device: torch.device) -> torch.Tensor:
    model.eval()
    context_length = int(getattr(model, "context_length", len(prefix_ids)))
    # Evaluation is next-token only, so if a prompt is longer than the model context,
    # use the most recent context window rather than crashing. The v4 calibration
    # config uses context_length=96, which covers the generated probes, but this
    # guard makes the pilot robust to future longer templates.
    prefix_ids = prefix_ids[-context_length:]
    ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(ids)["logits"][0, -1]
    return F.log_softmax(logits.float(), dim=-1).cpu()


@torch.no_grad()
def score_structural(model: torch.nn.Module, probes: list[dict[str, Any]], vocab: TextVocab, device: torch.device, limit: int | None = None) -> dict[str, Any]:
    rows = probes[:limit] if limit else probes
    margins: list[float] = []
    placement_margins: list[float] = []
    correct = 0
    by_template: dict[str, list[int]] = {}
    by_attractor: dict[str, list[int]] = {}
    for rec in rows:
        tokens = list(rec.get("tokens") or tokenize_text(rec["text"]))
        pos = int(rec["metric_targets"]["marker_value_margin"]["position"])
        good = rec["metric_targets"]["marker_value_margin"]["correct"]
        bad = rec["metric_targets"]["marker_value_margin"]["incorrect"]
        prefix = ["<bos>", *tokens[:pos]]
        prefix_ids = [vocab.token_to_id[t] for t in prefix]
        lps = next_token_logprobs(model, prefix_ids, device)
        margin = float(lps[vocab.token_to_id[good]] - lps[vocab.token_to_id[bad]])
        margins.append(margin)
        ok = int(margin > 0)
        correct += ok
        by_template.setdefault(str(rec.get("template", "unknown")), []).append(ok)
        by_attractor.setdefault(str(rec.get("attractor_number", "none")), []).append(ok)
        # Placement selectivity proxy. Compare correct marker probability at correct slot with
        # the same marker probability immediately after the verb, when that slot differs.
        verb_pos = int(rec.get("verb_index_transformed", 0)) + 1
        if verb_pos != pos and verb_pos < len(tokens):
            nohop_prefix = ["<bos>", *tokens[:verb_pos]]
            lps_nohop = next_token_logprobs(model, [vocab.token_to_id[t] for t in nohop_prefix], device)
            placement_margins.append(float(lps[vocab.token_to_id[good]] - lps_nohop[vocab.token_to_id[good]]))
    n = max(1, len(rows))
    out: dict[str, Any] = {
        "n": float(len(rows)),
        "accuracy": correct / n,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "median_margin": float(np.median(margins)) if margins else None,
        "placement_selectivity_proxy": float(np.mean(placement_margins)) if placement_margins else None,
    }
    for name, vals in by_template.items():
        out[f"acc_template_{name}"] = float(np.mean(vals))
    for name, vals in by_attractor.items():
        out[f"acc_attractor_{name}"] = float(np.mean(vals))
    return out


@torch.no_grad()
def score_facts(model: torch.nn.Module, probes: list[dict[str, Any]], vocab: TextVocab, device: torch.device, limit: int | None = None) -> dict[str, Any]:
    rows = probes[:limit] if limit else probes
    logps: list[float] = []
    by_depth: dict[str, list[float]] = {}
    for rec in rows:
        prompt_ids = vocab.encode(rec["prompt"], add_special=True)[:-1]
        target_tokens = tokenize_text(rec["target"])
        prefix = prompt_ids
        total = 0.0
        for tok in target_tokens:
            lps = next_token_logprobs(model, prefix, device)
            tid = vocab.token_to_id[tok]
            total += float(lps[tid])
            prefix = [*prefix, tid]
        avg = total / max(1, len(target_tokens))
        logps.append(avg)
        by_depth.setdefault(str(rec.get("depth", "unknown")), []).append(avg)
    out: dict[str, Any] = {
        "n": float(len(rows)),
        "mean_target_logprob_per_token": float(np.mean(logps)) if logps else None,
        "median_target_logprob_per_token": float(np.median(logps)) if logps else None,
    }
    for depth, vals in by_depth.items():
        out[f"mean_logprob_{depth}"] = float(np.mean(vals))
    return out


def train_steps(
    model: torch.nn.Module,
    dataset: PackedTextDataset,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    device: torch.device,
    precision: str,
    log_interval: int,
    log_path: Path,
    optimizer: torch.optim.Optimizer | None = None,
    global_step_offset: int = 0,
) -> dict[str, float]:
    # Keep optimizer state across chunks when an optimizer is provided.
    # This is important when training is segmented only to save checkpoints.
    opt = optimizer if optimizer is not None else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    started = time.time()
    ema = float("nan")
    last = float("nan")
    tokens_total = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for step in range(1, int(steps) + 1):
            abs_step = int(global_step_offset) + step
            cur_lr = linear_warmup_lr(abs_step, lr, warmup_steps)
            set_lr(opt, cur_lr)
            x, y, toks = dataset.make_batch(batch_size, device)
            opt.zero_grad(set_to_none=True)
            model.train()
            with maybe_autocast(device, precision):
                out = model(x, labels=y)
                loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.detach().cpu())
            ema = last if math.isnan(ema) else 0.98 * ema + 0.02 * last
            tokens_total += int(toks)
            if step == 1 or step % log_interval == 0 or step == steps:
                elapsed = time.time() - started
                rec = {"step": abs_step, "local_step": step, "loss": last, "ema_loss": ema, "lr": cur_lr, "elapsed_sec": elapsed, "tokens_total": tokens_total, "tokens_per_sec": tokens_total / max(elapsed, 1e-9)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
    elapsed = time.time() - started
    return {"last_loss": last, "ema_loss": ema, "elapsed_sec": elapsed, "tokens_per_sec": tokens_total / max(elapsed, 1e-9)}
