from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
import copy
import math
import random
import numpy as np
import torch
from torch import nn

from .data import VOCAB_SIZE, sample_toy_batch
from .model import TinyCausalTransformer, ToyModelConfig
from .metrics import evaluate_task, update_norms
from .spectral import model_spectral_summary
from .decodability import linear_probe_class_decodability


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def learning_rate_for_step(step: int, cfg: Dict[str, Any]) -> float:
    schedule = cfg.get("schedule", "s1_decay")
    peak = float(cfg.get("peak_lr", 3e-4))
    warmup = int(cfg.get("warmup_steps", 100))
    total = int(cfg.get("total_steps", 1000))
    min_lr = float(cfg.get("min_lr", 0.0))
    if step <= warmup:
        return peak * max(step, 1) / max(warmup, 1)
    if schedule in {"s2_constant", "constant"}:
        return float(cfg.get("constant_lr", peak))
    if schedule in {"s1_decay", "cosine_decay"}:
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_lr + (peak - min_lr) * cosine
    raise ValueError(f"Unknown schedule: {schedule}")


def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def build_model(cfg: Dict[str, Any]) -> TinyCausalTransformer:
    mcfg = cfg.get("model", {})
    return TinyCausalTransformer(
        ToyModelConfig(
            vocab_size=VOCAB_SIZE,
            d_model=int(mcfg.get("d_model", 64)),
            n_layers=int(mcfg.get("n_layers", 2)),
            n_heads=int(mcfg.get("n_heads", 4)),
            d_ff=int(mcfg.get("d_ff", 128)),
            max_seq_len=16,
            dropout=float(mcfg.get("dropout", 0.0)),
        )
    )


def train_base(cfg: Dict[str, Any], out_dir: Path, device: str) -> TinyCausalTransformer:
    set_seed(int(cfg.get("seed", 0)))
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg).to(device)
    train_cfg = cfg.get("base_train", {})
    total_steps = int(train_cfg.get("total_steps", 2000))
    batch_size = int(train_cfg.get("batch_size", 256))
    eval_every = int(train_cfg.get("eval_every", 200))
    ckpt_steps = set(int(x) for x in train_cfg.get("checkpoint_steps", [0, total_steps]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("peak_lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    train_log = out_dir / "metrics_base_train.jsonl"
    eval_log = out_dir / "metrics_base_eval.jsonl"
    train_log.write_text("")
    eval_log.write_text("")

    def save_ckpt(step: int):
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({"step": step, "model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg}, ckpt_dir / f"step_{step}.pt")

    def evaluate_and_log(step: int):
        row = {"step": step}
        row.update({f"base_{k}": v for k, v in evaluate_task(model, "base", device).items()})
        row.update({f"class_{k}": v for k, v in evaluate_task(model, "class_injection", device).items()})
        row.update(model_spectral_summary(model))
        row.update(linear_probe_class_decodability(model, device=device, steps=int(cfg.get("probe", {}).get("steps", 100)), lr=float(cfg.get("probe", {}).get("lr", 0.1)), seed=int(cfg.get("seed", 0)) + step))
        append_jsonl(eval_log, row)

    if 0 in ckpt_steps:
        save_ckpt(0)
    evaluate_and_log(0)
    model.train()
    for step in range(1, total_steps + 1):
        lr = learning_rate_for_step(step, train_cfg)
        set_optimizer_lr(opt, lr)
        batch = sample_toy_batch(batch_size, task="base", device=device)
        loss = model.loss(batch.input_ids, batch.labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
        opt.step()
        if step % int(train_cfg.get("log_every", 50)) == 0 or step == 1:
            append_jsonl(train_log, {"step": step, "loss": float(loss.item()), "lr": lr})
        if step % eval_every == 0 or step in ckpt_steps or step == total_steps:
            evaluate_and_log(step)
        if step in ckpt_steps or step == total_steps:
            save_ckpt(step)
    return model


def load_model_from_ckpt(path: Path, cfg: Dict[str, Any], device: str) -> TinyCausalTransformer:
    model = build_model(cfg).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    return model


def run_injection_cell(
    base_ckpt: Path,
    cfg: Dict[str, Any],
    out_dir: Path,
    device: str,
    *,
    checkpoint_step: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inj_cfg = cfg.get("injection", {})
    model = load_model_from_ckpt(base_ckpt, cfg, device)
    before_params = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    pre = evaluate_task(model, "class_injection", device)
    pre_base = evaluate_task(model, "base", device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(inj_cfg.get("lr", 3e-4)), weight_decay=float(inj_cfg.get("weight_decay", 0.0)))
    metrics_path = out_dir / "metrics_injection.jsonl"
    metrics_path.write_text("")
    batch_size = int(inj_cfg.get("batch_size", 256))
    for step in range(1, int(inj_cfg.get("steps", 300)) + 1):
        model.train()
        batch = sample_toy_batch(batch_size, task="class_injection", device=device)
        loss = model.loss(batch.input_ids, batch.labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(inj_cfg.get("grad_clip", 1.0)))
        opt.step()
        if step % int(inj_cfg.get("log_every", 50)) == 0 or step == 1:
            append_jsonl(metrics_path, {"step": step, "loss": float(loss.item())})
    post = evaluate_task(model, "class_injection", device)
    post_base = evaluate_task(model, "base", device)
    wash = None
    if int(cfg.get("washout", {}).get("steps", 0)) > 0:
        wcfg = cfg.get("washout", {})
        opt_w = torch.optim.AdamW(model.parameters(), lr=float(wcfg.get("lr", inj_cfg.get("lr", 3e-4))), weight_decay=float(wcfg.get("weight_decay", 0.0)))
        wpath = out_dir / "metrics_washout.jsonl"
        wpath.write_text("")
        for step in range(1, int(wcfg.get("steps", 100)) + 1):
            model.train()
            batch = sample_toy_batch(batch_size, task="base", device=device)
            loss = model.loss(batch.input_ids, batch.labels)
            opt_w.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(wcfg.get("grad_clip", 1.0)))
            opt_w.step()
            if step % int(wcfg.get("log_every", 50)) == 0 or step == 1:
                append_jsonl(wpath, {"step": step, "loss": float(loss.item())})
        wash = evaluate_task(model, "class_injection", device)
    norms = update_norms(before_params, model)
    row = {
        "checkpoint_step": checkpoint_step,
        "pre_accuracy": pre["accuracy"],
        "post_accuracy": post["accuracy"],
        "uptake": post["accuracy"] - pre["accuracy"],
        "pre_margin": pre["margin"],
        "post_margin": post["margin"],
        "margin_uptake": post["margin"] - pre["margin"],
        "base_accuracy_pre": pre_base["accuracy"],
        "base_accuracy_post": post_base["accuracy"],
    }
    if wash is not None:
        row.update({"retention_accuracy": wash["accuracy"], "retention_delta": wash["accuracy"] - pre["accuracy"], "retained_fraction": ((wash["accuracy"] - pre["accuracy"]) / (post["accuracy"] - pre["accuracy"] + 1e-12))})
    row.update(norms)
    (out_dir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    return row
