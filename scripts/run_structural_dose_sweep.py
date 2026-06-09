#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from cplm.pilot.textlm import (
    MarkerOnlyDataset,
    PackedTextDataset,
    TextVocab,
    build_tiny_model,
    choose_device,
    load_jsonl,
    score_structural,
    train_steps,
)


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(obj), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_id(name: str) -> str:
    return f"{name}__{time.strftime('%Y%m%d_%H%M%S')}"


def load_structural(dataset_dir: Path, arm: str, split: str) -> list[dict[str, Any]]:
    return load_jsonl(dataset_dir / f"{arm}_{split}.jsonl")


def fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    seed = int(exp.get("seed", 0))
    torch.manual_seed(seed)
    device = choose_device(str(exp.get("device", "cuda")))
    if device.type == "cpu":
        torch.set_num_threads(int(exp.get("cpu_threads", 1)))
    precision = str(exp.get("precision", "bf16"))

    dataset_dir = Path(exp["dataset_dir"])
    output_root = Path(exp.get("output_root", "results/fast_pilot"))
    run_id = make_run_id(str(exp.get("name", "fast_structural_dose_sweep_v4")))
    out_dir = output_root / run_id
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    dump_json(cfg, out_dir / "config_resolved.json")

    max_train = int(exp.get("max_train_records_per_arm", 2000))
    max_probe = int(exp.get("max_probe_records_per_arm", 300))
    wordhop_train = load_structural(dataset_dir, "wordhop", "train")[:max_train]
    nohop_train = load_structural(dataset_dir, "nohop", "train")[:max_train]
    wordhop_probe = load_structural(dataset_dir, "wordhop", "probe")[:max_probe]
    nohop_probe = load_structural(dataset_dir, "nohop", "probe")[:max_probe]

    # Use unmarked source text as the tiny naturalish base stream.
    seen: set[str] = set()
    base_texts: list[str] = []
    for rec in wordhop_train + nohop_train:
        text = rec.get("source_text") or rec["text"]
        if text not in seen:
            seen.add(text)
            base_texts.append(text)

    all_texts: list[str] = []
    all_texts.extend(base_texts)
    all_texts.extend([r["text"] for r in wordhop_train])
    all_texts.extend([r["text"] for r in nohop_train])
    all_texts.extend([r["text"] for r in wordhop_probe])
    all_texts.extend([r["text"] for r in nohop_probe])
    vocab = TextVocab.build(all_texts)
    dump_json(vocab.to_jsonable(), out_dir / "vocab.json")

    ctx = int(cfg["model"]["context_length"])
    base_ds = PackedTextDataset([vocab.encode(t) for t in base_texts], vocab.pad_id, ctx, seed=seed)
    marker_only_loss = bool(cfg.get("sweep", {}).get("marker_only_loss", False))
    arm_datasets = {
        "wordhop": (
            MarkerOnlyDataset(wordhop_train, vocab, vocab.pad_id, ctx, seed=seed + 11)
            if marker_only_loss
            else PackedTextDataset([vocab.encode(r["text"]) for r in wordhop_train], vocab.pad_id, ctx, seed=seed + 11)
        ),
        "nohop": (
            MarkerOnlyDataset(nohop_train, vocab, vocab.pad_id, ctx, seed=seed + 12)
            if marker_only_loss
            else PackedTextDataset([vocab.encode(r["text"]) for r in nohop_train], vocab.pad_id, ctx, seed=seed + 12)
        ),
    }
    probes = {"wordhop": wordhop_probe, "nohop": nohop_probe}

    # Base training once.
    model = build_tiny_model(cfg, len(vocab.id_to_token)).to(device)
    base_cfg = cfg["base_train"]
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(base_cfg["lr"]),
        weight_decay=float(base_cfg.get("weight_decay", 0.0)),
    )
    base_stats = train_steps(
        model,
        base_ds,
        steps=int(base_cfg["steps"]),
        batch_size=int(base_cfg["batch_size"]),
        lr=float(base_cfg["lr"]),
        weight_decay=float(base_cfg.get("weight_decay", 0.0)),
        warmup_steps=int(base_cfg.get("warmup_steps", 0)),
        device=device,
        precision=precision,
        log_interval=int(base_cfg.get("log_interval", 20)),
        log_path=out_dir / "metrics_base_train.jsonl",
        optimizer=base_optimizer,
        global_step_offset=0,
    )
    ckpt_step = int(base_cfg.get("checkpoint", base_cfg["steps"]))
    ckpt_dir = out_dir / "base_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": base_optimizer.state_dict(), "step": ckpt_step, "stats": base_stats}, ckpt_dir / f"step_{ckpt_step}.pt")
    print(f"BASE checkpoint step={ckpt_step} loss={base_stats['ema_loss']:.4f} tok/s={base_stats['tokens_per_sec']:.0f}", flush=True)

    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sweep = cfg["sweep"]
    arms = list(sweep.get("arms", ["nohop", "wordhop"]))
    lrs = [float(x) for x in sweep.get("lrs", [1e-3])]
    step_grid = [int(x) for x in sweep.get("steps", [300, 600, 1000, 1500])]
    summary_rows: list[dict[str, Any]] = []

    for lr in lrs:
        for steps in step_grid:
            for arm in arms:
                cell_id = f"step_{ckpt_step}_{arm}_lr_{fmt_lr(lr)}_n_{steps}"
                cell_dir = out_dir / "cells" / cell_id
                cell_dir.mkdir(parents=True, exist_ok=True)
                cell_model = build_tiny_model(cfg, len(vocab.id_to_token)).to(device)
                cell_model.load_state_dict(base_state)
                pre = score_structural(cell_model, probes[arm], vocab, device, max_probe)
                dump_json(pre, cell_dir / "pre.json")
                inj_stats = train_steps(
                    cell_model,
                    arm_datasets[arm],
                    steps=steps,
                    batch_size=int(sweep["batch_size"]),
                    lr=lr,
                    weight_decay=float(sweep.get("weight_decay", 0.0)),
                    warmup_steps=max(1, int(round(float(sweep.get("warmup_fraction", 0.05)) * steps))),
                    device=device,
                    precision=precision,
                    log_interval=int(sweep.get("log_interval", 100)),
                    log_path=cell_dir / "metrics_injection.jsonl",
                )
                post = score_structural(cell_model, probes[arm], vocab, device, max_probe)
                dump_json(post, cell_dir / "uptake.json")
                row: dict[str, Any] = {
                    "checkpoint_step": ckpt_step,
                    "arm": arm,
                    "lr": lr,
                    "steps": steps,
                    "pre": pre.get("accuracy"),
                    "post": post.get("accuracy"),
                    "uptake": post.get("accuracy", 0.0) - pre.get("accuracy", 0.0),
                    "mean_margin_pre": pre.get("mean_margin"),
                    "mean_margin_post": post.get("mean_margin"),
                    "placement_selectivity_pre": pre.get("placement_selectivity_proxy"),
                    "placement_selectivity_post": post.get("placement_selectivity_proxy"),
                    "inj_elapsed_sec": inj_stats["elapsed_sec"],
                    "inj_last_loss": inj_stats["last_loss"],
                    "inj_ema_loss": inj_stats["ema_loss"],
                }
                # Include key subgroup accuracies for quick inspection.
                for k, v in post.items():
                    if k.startswith("acc_attractor_") or k.startswith("acc_template_"):
                        row[f"post_{k}"] = v
                summary_rows.append(row)
                print(f"CELL {cell_id} pre={row['pre']:.4f} post={row['post']:.4f} uptake={row['uptake']:.4f}", flush=True)

    summary_path = out_dir / "summary.csv"
    keys = sorted({k for r in summary_rows for k in r.keys()})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    target = sweep.get("target", {})
    nohop_min = float(target.get("nohop_min", 0.70))
    nohop_max = float(target.get("nohop_max", 0.95))
    wordhop_min = float(target.get("wordhop_min", 0.60))
    wordhop_max = float(target.get("wordhop_max", 0.90))
    by_dose: dict[tuple[float, int], dict[str, float]] = {}
    for r in summary_rows:
        by_dose.setdefault((float(r["lr"]), int(r["steps"])), {})[str(r["arm"])] = float(r["post"])
    candidates: list[tuple[float, int, float, float]] = []
    for (lr, steps), vals in by_dose.items():
        n = vals.get("nohop")
        w = vals.get("wordhop")
        if n is None or w is None:
            continue
        if nohop_min <= n <= nohop_max and wordhop_min <= w <= wordhop_max:
            # Prefer middle of range; smaller dose if tied.
            center_penalty = abs(n - (nohop_min + nohop_max) / 2) + abs(w - (wordhop_min + wordhop_max) / 2)
            candidates.append((center_penalty, lr, steps, w))
    candidates.sort(key=lambda x: (x[0], x[2], x[1]))

    report: list[str] = []
    report.append("# Structural dose sweep v4 report")
    report.append("")
    report.append(f"Run id: `{run_id}`")
    report.append("")
    report.append("This is a tiny-model calibration for choosing a non-ceiling structural injection dose before BabyLM-scale pilots. It is not final thesis evidence.")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("| lr | steps | NOHOP post | WORDHOP post | decision |")
    report.append("| ---: | ---: | ---: | ---: | --- |")
    for lr in lrs:
        for steps in step_grid:
            vals = by_dose.get((lr, steps), {})
            n = vals.get("nohop")
            w = vals.get("wordhop")
            if n is None or w is None:
                decision = "missing"
            elif n >= 0.98 and w >= 0.98:
                decision = "ceiling diagnostic"
            elif nohop_min <= n <= nohop_max and wordhop_min <= w <= wordhop_max:
                decision = "candidate"
            elif n < nohop_min:
                decision = "underpowered"
            elif w < wordhop_min:
                decision = "WORDHOP weak"
            elif n > nohop_max or w > wordhop_max:
                decision = "too strong / near ceiling"
            else:
                decision = "inspect"
            report.append(f"| {lr:g} | {steps} | {n if n is not None else 'NA'} | {w if w is not None else 'NA'} | {decision} |")
    report.append("")
    report.append("## Automatic recommendation")
    report.append("")
    if candidates:
        _, lr, steps, _ = candidates[0]
        vals = by_dose[(lr, steps)]
        report.append(f"Recommended initial BabyLM-pilot dose: `lr={lr:g}`, `steps={steps}`. NOHOP post={vals['nohop']:.4f}, WORDHOP post={vals['wordhop']:.4f}.")
    else:
        report.append("No cell landed inside the target non-ceiling band. If low-dose cells are underpowered and high-dose cells saturate, run a finer sweep around the transition, for example `lr=6e-4` with steps `[600, 1000, 1500]`.")
    report.append("")
    report.append("Target band:")
    report.append(f"- NOHOP: {nohop_min:.2f} to {nohop_max:.2f}")
    report.append(f"- WORDHOP: {wordhop_min:.2f} to {wordhop_max:.2f}")
    (out_dir / "dose_sweep_report.md").write_text("\n".join(report), encoding="utf-8")
    dump_json({
        "run_id": run_id,
        "output_dir": str(out_dir),
        "device": str(device),
        "n_vocab": len(vocab.id_to_token),
        "n_summary_rows": len(summary_rows),
        "base_checkpoint_step": ckpt_step,
        "candidate_count": len(candidates),
    }, out_dir / "manifest.json")
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
