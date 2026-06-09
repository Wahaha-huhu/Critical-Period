#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
import torch

from cplm.pilot.textlm import (
    PackedTextDataset,
    TextVocab,
    build_tiny_model,
    choose_device,
    load_jsonl,
    score_facts,
    score_structural,
    train_steps,
)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_id(name: str) -> str:
    return f"{name}__{time.strftime('%Y%m%d_%H%M%S')}"


def structural_records(dataset_dir: Path, arm: str, split: str) -> list[dict[str, Any]]:
    return load_jsonl(dataset_dir / f"{arm}_{split}.jsonl")


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
    run_id = make_run_id(str(exp.get("name", "fast_injection_pilot")))
    out_dir = output_root / run_id
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    dump_json(cfg, out_dir / "config_resolved.json")

    max_train = int(exp.get("max_train_records_per_arm", 600))
    max_probe = int(exp.get("max_probe_records_per_arm", 120))
    wordhop_train = structural_records(dataset_dir, "wordhop", "train")[:max_train]
    nohop_train = structural_records(dataset_dir, "nohop", "train")[:max_train]
    facts_train = load_jsonl(dataset_dir / "facts_train.jsonl")[:max_train]
    wordhop_probe = structural_records(dataset_dir, "wordhop", "probe")[:max_probe]
    nohop_probe = structural_records(dataset_dir, "nohop", "probe")[:max_probe]
    facts_probe = load_jsonl(dataset_dir / "facts_probe.jsonl")[:max_probe]

    # Naturalish base stream uses original English source text from structural records only.
    base_texts = []
    seen = set()
    for rec in wordhop_train + nohop_train:
        text = rec.get("source_text") or rec["text"]
        if text not in seen:
            seen.add(text)
            base_texts.append(text)

    all_texts = []
    all_texts.extend(base_texts)
    all_texts.extend([r["text"] for r in wordhop_train])
    all_texts.extend([r["text"] for r in nohop_train])
    all_texts.extend([r["text"] for r in facts_train])
    all_texts.extend([r["text"] for r in wordhop_probe])
    all_texts.extend([r["text"] for r in nohop_probe])
    all_texts.extend([r["prompt"] + " " + r["target"] for r in facts_probe])
    vocab = TextVocab.build(all_texts)
    dump_json(vocab.to_jsonable(), out_dir / "vocab.json")

    ctx = int(cfg["model"]["context_length"])
    base_ds = PackedTextDataset([vocab.encode(t) for t in base_texts], vocab.pad_id, ctx, seed=seed)
    arm_datasets = {
        "wordhop": PackedTextDataset([vocab.encode(r["text"]) for r in wordhop_train], vocab.pad_id, ctx, seed=seed + 11),
        "nohop": PackedTextDataset([vocab.encode(r["text"]) for r in nohop_train], vocab.pad_id, ctx, seed=seed + 12),
        "facts": PackedTextDataset([vocab.encode(r["text"]) for r in facts_train], vocab.pad_id, ctx, seed=seed + 13),
    }

    model = build_tiny_model(cfg, len(vocab.id_to_token)).to(device)
    base_cfg = cfg["base_train"]
    checkpoints = set(int(x) for x in base_cfg.get("checkpoints", []))
    ckpt_dir = out_dir / "base_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    if 0 in checkpoints:
        torch.save({"model": model.state_dict(), "step": 0}, ckpt_dir / "step_0.pt")

    # Manual base loop so we can save checkpoints.
    total_steps = int(base_cfg["steps"])
    chunk_start = 1
    last_saved = 0
    for ckpt in sorted([c for c in checkpoints if c > 0]):
        steps_to_run = ckpt - last_saved
        if steps_to_run <= 0:
            continue
        stats = train_steps(
            model,
            base_ds,
            steps=steps_to_run,
            batch_size=int(base_cfg["batch_size"]),
            lr=float(base_cfg["lr"]),
            weight_decay=float(base_cfg.get("weight_decay", 0.0)),
            warmup_steps=max(0, int(base_cfg.get("warmup_steps", 0)) - last_saved),
            device=device,
            precision=precision,
            log_interval=int(base_cfg.get("log_interval", 20)),
            log_path=out_dir / "metrics_base_train.jsonl",
        )
        last_saved = ckpt
        torch.save({"model": model.state_dict(), "step": ckpt, "stats": stats}, ckpt_dir / f"step_{ckpt}.pt")
        print(f"BASE checkpoint step={ckpt} loss={stats['ema_loss']:.4f} tok/s={stats['tokens_per_sec']:.0f}", flush=True)

    arms = list(cfg["injection"].get("arms", ["wordhop", "nohop", "facts"]))
    pilot_ckpts = [int(x) for x in cfg["injection"].get("checkpoints", sorted(checkpoints))]
    for ckpt in pilot_ckpts:
        state = torch.load(ckpt_dir / f"step_{ckpt}.pt", map_location=device)
        for arm in arms:
            cell_id = f"step_{ckpt}_{arm}"
            cell_dir = out_dir / "cells" / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_model = build_tiny_model(cfg, len(vocab.id_to_token)).to(device)
            cell_model.load_state_dict(state["model"])
            if arm == "facts":
                pre = score_facts(cell_model, facts_probe, vocab, device, max_probe)
            elif arm == "wordhop":
                pre = score_structural(cell_model, wordhop_probe, vocab, device, max_probe)
            elif arm == "nohop":
                pre = score_structural(cell_model, nohop_probe, vocab, device, max_probe)
            else:
                raise ValueError(f"unknown arm {arm}")
            dump_json(pre, cell_dir / "pre.json")

            inj_cfg = cfg["injection"]
            inj_stats = train_steps(
                cell_model,
                arm_datasets[arm],
                steps=int(inj_cfg["steps"]),
                batch_size=int(inj_cfg["batch_size"]),
                lr=float(inj_cfg["lr"]),
                weight_decay=float(inj_cfg.get("weight_decay", 0.0)),
                warmup_steps=max(1, int(round(float(inj_cfg.get("warmup_fraction", 0.15)) * int(inj_cfg["steps"])))),
                device=device,
                precision=precision,
                log_interval=int(inj_cfg.get("log_interval", 10)),
                log_path=cell_dir / "metrics_injection.jsonl",
            )
            if arm == "facts":
                post = score_facts(cell_model, facts_probe, vocab, device, max_probe)
            elif arm == "wordhop":
                post = score_structural(cell_model, wordhop_probe, vocab, device, max_probe)
            else:
                post = score_structural(cell_model, nohop_probe, vocab, device, max_probe)
            dump_json(post, cell_dir / "uptake.json")
            retention = None
            washout_stats = None
            if bool(cfg.get("washout", {}).get("enabled", True)):
                wash_cfg = cfg["washout"]
                washout_stats = train_steps(
                    cell_model,
                    base_ds,
                    steps=int(wash_cfg["steps"]),
                    batch_size=int(wash_cfg["batch_size"]),
                    lr=float(wash_cfg["lr"]),
                    weight_decay=float(wash_cfg.get("weight_decay", 0.0)),
                    warmup_steps=0,
                    device=device,
                    precision=precision,
                    log_interval=int(wash_cfg.get("log_interval", 20)),
                    log_path=cell_dir / "metrics_washout.jsonl",
                )
                if arm == "facts":
                    retention = score_facts(cell_model, facts_probe, vocab, device, max_probe)
                elif arm == "wordhop":
                    retention = score_structural(cell_model, wordhop_probe, vocab, device, max_probe)
                else:
                    retention = score_structural(cell_model, nohop_probe, vocab, device, max_probe)
                dump_json(retention, cell_dir / "retention.json")
            row = {"checkpoint_step": ckpt, "arm": arm}
            if arm == "facts":
                metric = "mean_target_logprob_per_token"
                row.update({
                    "pre": pre.get(metric),
                    "post": post.get(metric),
                    "uptake": (post.get(metric, 0.0) - pre.get(metric, 0.0)),
                    "retention": None if retention is None else retention.get(metric),
                    "retention_delta": None if retention is None else retention.get(metric, 0.0) - pre.get(metric, 0.0),
                })
            else:
                metric = "accuracy"
                row.update({
                    "pre": pre.get(metric),
                    "post": post.get(metric),
                    "uptake": (post.get(metric, 0.0) - pre.get(metric, 0.0)),
                    "retention": None if retention is None else retention.get(metric),
                    "retention_delta": None if retention is None else retention.get(metric, 0.0) - pre.get(metric, 0.0),
                    "placement_selectivity_pre": pre.get("placement_selectivity_proxy"),
                    "placement_selectivity_post": post.get("placement_selectivity_proxy"),
                })
            row["inj_elapsed_sec"] = inj_stats["elapsed_sec"]
            row["washout_elapsed_sec"] = None if washout_stats is None else washout_stats["elapsed_sec"]
            summary_rows.append(row)
            print(f"CELL {cell_id} pre={row['pre']:.4f} post={row['post']:.4f} ret={row.get('retention')} ", flush=True)
    # Write summary csv and report.
    import csv
    summary_path = out_dir / "summary.csv"
    keys = sorted({k for r in summary_rows for k in r.keys()})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    report = ["# Fast injection pilot report", "", f"Run id: `{run_id}`", "", "## Summary", ""]
    report.append("| checkpoint | arm | pre | post | uptake | retention | retention_delta |")
    report.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        report.append(f"| {r['checkpoint_step']} | {r['arm']} | {r.get('pre', float('nan')):.4f} | {r.get('post', float('nan')):.4f} | {r.get('uptake', float('nan')):.4f} | {r.get('retention') if r.get('retention') is not None else 'NA'} | {r.get('retention_delta') if r.get('retention_delta') is not None else 'NA'} |")
    report.append("")
    report.append("## Interpretation gate")
    report.append("")
    report.append("This fast pilot only checks the training, checkpoint, scoring, injection, and washout path. It is not the final BabyLM evidence.")
    (out_dir / "pilot_report.md").write_text("\n".join(report), encoding="utf-8")
    dump_json({"run_id": run_id, "output_dir": str(out_dir), "device": str(device), "n_vocab": len(vocab.id_to_token), "n_summary_rows": len(summary_rows)}, out_dir / "manifest.json")
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
