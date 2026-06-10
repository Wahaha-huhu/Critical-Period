from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.optim import AdamW

from .data import InductionVocab, make_batch, sequence_length
from .metrics import evaluate_recall, model_spectral_summary
from .model import InductionModelConfig, InductionTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at_step(step: int, cfg: Dict) -> float:
    """Learning-rate schedule used by the induction toy.

    S1 is the standard warmup + cosine decay schedule.
    S2 is warmup + constant post-warmup rate.
    S3 is warmup + cosine-restart cycles, used as a plasticity-restoration
    counterfactual. All schedules share the same optimiser, weight decay, data,
    batch size, and total training length; only the post-warmup LR trajectory
    changes.
    """
    sched = cfg.get("schedule", "s1_decay")
    warmup = int(cfg.get("warmup_steps", 0))
    peak_lr = float(cfg.get("peak_lr", cfg.get("lr", 5e-4)))
    total = int(cfg.get("total_steps", 1000))
    if warmup > 0 and step <= warmup:
        return peak_lr * step / max(1, warmup)
    if sched == "s1_decay":
        min_lr = float(cfg.get("min_lr", peak_lr * 0.02))
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine
    if sched == "s2_constant":
        return float(cfg.get("constant_lr", peak_lr))
    if sched in {"s3_cyclic", "s3_restart", "cyclic_restart"}:
        min_lr = float(cfg.get("min_lr", peak_lr * 0.05))
        max_lr = float(cfg.get("max_lr", peak_lr))
        cycle_steps = int(cfg.get("cycle_steps", max(1, (total - warmup) // 4)))
        # Cosine-restart cycle: high immediately after each restart and low just
        # before the next restart. This is intentionally simple and deterministic.
        phase = (step - warmup) % max(1, cycle_steps)
        progress = phase / max(1, cycle_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine
    raise ValueError(f"Unknown schedule: {sched}")


def build_model_and_vocab(cfg: Dict, device: str | torch.device):
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    vocab = InductionVocab(n_symbols=int(data_cfg.get("n_symbols", 64)))
    n_pairs = int(data_cfg.get("n_pairs", 8))
    n_queries = int(data_cfg.get("n_queries", 1))
    if str(data_cfg.get("sequence_mode", "query")) == "copy_repeat":
        n_queries = max(1, n_pairs - 1)
    max_seq_len = max(int(model_cfg.get("max_seq_len", 0) or 0), sequence_length(n_pairs, n_queries))
    mcfg = InductionModelConfig(
        vocab_size=vocab.vocab_size,
        d_model=int(model_cfg.get("d_model", 96)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        d_ff=int(model_cfg.get("d_ff", 192)),
        max_seq_len=max_seq_len,
        dropout=float(model_cfg.get("dropout", 0.0)),
        tie_embeddings=bool(model_cfg.get("tie_embeddings", True)),
    )
    model = InductionTransformer(mcfg).to(device)
    return model, vocab


def train_induction_base(cfg: Dict, out_dir: Path, device: str | torch.device) -> None:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    model, vocab = build_model_and_vocab(cfg, device)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("base_train", {})
    eval_cfg = cfg.get("eval", {})
    n_pairs = int(data_cfg.get("n_pairs", 8))
    n_queries = int(data_cfg.get("n_queries", 1))
    if str(data_cfg.get("sequence_mode", "query")) == "copy_repeat":
        n_queries = max(1, n_pairs - 1)
    batch_size = int(train_cfg.get("batch_size", 64))
    total_steps = int(train_cfg.get("total_steps", 1000))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    eval_every = int(train_cfg.get("eval_every", 100))
    checkpoint_steps = set(int(x) for x in train_cfg.get("checkpoint_steps", []))
    checkpoint_steps.add(0)
    checkpoint_steps.add(total_steps)

    opt = AdamW(model.parameters(), lr=float(train_cfg.get("peak_lr", 5e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log = (out_dir / "metrics_train.jsonl").open("w", encoding="utf-8")
    eval_log = (out_dir / "metrics_eval.jsonl").open("w", encoding="utf-8")

    def save_ckpt(step: int) -> None:
        torch.save({"step": step, "model": model.state_dict(), "config": cfg}, ckpt_dir / f"step_{step}.pt")

    def run_eval(step: int) -> None:
        metrics = evaluate_recall(
            model,
            vocab=vocab,
            n_pairs=n_pairs,
            n_queries=n_queries,
            batch_size=int(eval_cfg.get("batch_size", 256)),
            n_batches=int(eval_cfg.get("n_batches", 4)),
            device=device,
            role_mode=str(data_cfg.get("role_mode", "shared")),
            sequence_mode=str(data_cfg.get("sequence_mode", "query")),
        )
        metrics.update(model_spectral_summary(model))
        metrics["step"] = step
        eval_log.write(json.dumps(metrics, sort_keys=True) + "\n")
        eval_log.flush()

    save_ckpt(0)
    run_eval(0)
    model.train()
    rng = random.Random(seed + 17)
    for step in range(1, total_steps + 1):
        lr = lr_at_step(step, train_cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        batch = make_batch(batch_size=batch_size, n_pairs=n_pairs, n_queries=n_queries, vocab=vocab, device=device, rng=rng, role_mode=str(data_cfg.get("role_mode", "shared")), sequence_mode=str(data_cfg.get("sequence_mode", "query")))
        loss = model.loss(batch["input_ids"], batch["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step % log_every == 0 or step == 1:
            row = {"step": step, "train_loss": float(loss.detach().cpu()), "lr": lr}
            train_log.write(json.dumps(row, sort_keys=True) + "\n")
            train_log.flush()
        if step % eval_every == 0 or step in checkpoint_steps:
            run_eval(step)
        if step in checkpoint_steps:
            save_ckpt(step)

    train_log.close()
    eval_log.close()
