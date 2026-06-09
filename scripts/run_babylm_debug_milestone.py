#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
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
    maybe_autocast,
    score_facts,
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


def _chunk_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Prefer breaking at whitespace.
        if end < len(text):
            ws = text.rfind(" ", start, end)
            if ws > start + max_chars // 2:
                end = ws
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def read_corpus_texts(corpus_cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    patterns = list(corpus_cfg.get("globs", []) or [])
    max_docs = int(corpus_cfg.get("max_documents", 20000))
    max_chars = int(corpus_cfg.get("max_chars_per_document", 20000))
    jsonl_field = str(corpus_cfg.get("jsonl_text_field", "text"))
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(str(pat), recursive=True))
    paths = sorted({p.resolve() for p in paths if p.is_file()})
    texts: list[str] = []
    for path in paths:
        if len(texts) >= max_docs:
            break
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if len(texts) >= max_docs:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        val = obj.get(jsonl_field)
                        if isinstance(val, str):
                            for chunk in _chunk_text(val, max_chars):
                                texts.append(chunk)
                                if len(texts) >= max_docs:
                                    break
            else:
                raw = path.read_text(encoding="utf-8", errors="ignore")
                # Treat nonempty lines as documents when possible; otherwise chunk the whole file.
                lines = [x.strip() for x in raw.splitlines() if x.strip()]
                candidates = lines if len(lines) > 1 else [raw]
                for cand in candidates:
                    if len(texts) >= max_docs:
                        break
                    for chunk in _chunk_text(cand, max_chars):
                        texts.append(chunk)
                        if len(texts) >= max_docs:
                            break
        except OSError:
            continue
    meta = {"patterns": patterns, "matched_files": len(paths), "documents_loaded": len(texts)}
    return texts, meta


def split_train_val(texts: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    xs = list(texts)
    rng.shuffle(xs)
    if len(xs) < 2 or val_fraction <= 0:
        return xs, xs[:]
    n_val = max(1, int(round(len(xs) * val_fraction)))
    n_val = min(n_val, len(xs) - 1)
    return xs[n_val:], xs[:n_val]


def load_structural(dataset_dir: Path, arm: str, split: str) -> list[dict[str, Any]]:
    return load_jsonl(dataset_dir / f"{arm}_{split}.jsonl")


def collect_fallback_sources(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for rec in records:
        text = rec.get("source_text") or rec["text"]
        if text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def eval_loss(model: torch.nn.Module, dataset: PackedTextDataset, *, batches: int, batch_size: int, device: torch.device, precision: str) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    toks_total = 0
    for _ in range(int(batches)):
        x, y, toks = dataset.make_batch(batch_size, device)
        with maybe_autocast(device, precision):
            out = model(x, labels=y)
        losses.append(float(out["loss"].detach().cpu()))
        toks_total += int(toks)
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return {"loss": mean_loss, "ppl": float(math.exp(min(mean_loss, 20.0))), "tokens": float(toks_total), "batches": float(batches)}


def write_eval_record(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_json(rec), allow_nan=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    seed = int(exp.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(str(exp.get("device", "cuda")))
    if device.type == "cpu":
        torch.set_num_threads(int(exp.get("cpu_threads", 1)))
    precision = str(exp.get("precision", "bf16"))

    dataset_dir = Path(exp["dataset_dir"])
    output_root = Path(exp.get("output_root", "results/babylm_debug"))
    run_id = make_run_id(str(exp.get("name", "babylm_debug_milestone")))
    out_dir = output_root / run_id
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    dump_json(cfg, out_dir / "config_resolved.json")

    max_train = int(exp.get("max_train_records_per_arm", 2000))
    max_probe = int(exp.get("max_probe_records_per_arm", 300))
    wordhop_train = load_structural(dataset_dir, "wordhop", "train")[:max_train]
    nohop_train = load_structural(dataset_dir, "nohop", "train")[:max_train]
    facts_train = load_jsonl(dataset_dir / "facts_train.jsonl")[:max_train]
    wordhop_probe = load_structural(dataset_dir, "wordhop", "probe")[:max_probe]
    nohop_probe = load_structural(dataset_dir, "nohop", "probe")[:max_probe]
    facts_probe = load_jsonl(dataset_dir / "facts_probe.jsonl")[:max_probe]

    corpus_cfg = cfg.get("corpus", {})
    corpus_texts, corpus_meta = read_corpus_texts(corpus_cfg)
    used_fallback = False
    if not corpus_texts:
        if not bool(corpus_cfg.get("fallback_to_injection_sources", True)):
            raise RuntimeError("No corpus files matched and fallback_to_injection_sources=false")
        corpus_texts = collect_fallback_sources(wordhop_train + nohop_train)
        used_fallback = True
    train_texts, val_texts = split_train_val(corpus_texts, float(corpus_cfg.get("validation_fraction", 0.05)), seed)

    # Include injection/probe strings in the vocab so all scored tokens are known.
    vocab_texts: list[str] = []
    vocab_texts.extend(train_texts)
    vocab_texts.extend(val_texts)
    vocab_texts.extend(r["text"] for r in wordhop_train)
    vocab_texts.extend(r["text"] for r in nohop_train)
    vocab_texts.extend(r["text"] for r in facts_train)
    vocab_texts.extend(r["text"] for r in wordhop_probe)
    vocab_texts.extend(r["text"] for r in nohop_probe)
    vocab_texts.extend(r["prompt"] + " " + r["target"] for r in facts_probe)
    vocab = TextVocab.build(vocab_texts)
    dump_json(vocab.to_jsonable(), out_dir / "vocab.json")

    ctx = int(cfg["model"]["context_length"])
    train_ds = PackedTextDataset([vocab.encode(t) for t in train_texts], vocab.pad_id, ctx, seed=seed)
    val_ds = PackedTextDataset([vocab.encode(t) for t in val_texts], vocab.pad_id, ctx, seed=seed + 1)
    marker_only = bool(cfg.get("injection", {}).get("marker_only_loss", False))
    arm_datasets = {
        "wordhop": MarkerOnlyDataset(wordhop_train, vocab, vocab.pad_id, ctx, seed=seed + 11) if marker_only else PackedTextDataset([vocab.encode(r["text"]) for r in wordhop_train], vocab.pad_id, ctx, seed=seed + 11),
        "nohop": MarkerOnlyDataset(nohop_train, vocab, vocab.pad_id, ctx, seed=seed + 12) if marker_only else PackedTextDataset([vocab.encode(r["text"]) for r in nohop_train], vocab.pad_id, ctx, seed=seed + 12),
        "facts": PackedTextDataset([vocab.encode(r["text"]) for r in facts_train], vocab.pad_id, ctx, seed=seed + 13),
    }

    model = build_tiny_model(cfg, len(vocab.id_to_token)).to(device)
    n_params = count_parameters(model)
    bcfg = cfg["backbone_train"]
    checkpoints = sorted({int(x) for x in bcfg.get("checkpoints", [])} | {int(bcfg["steps"])})
    total_steps = int(bcfg["steps"])
    eval_interval = int(bcfg.get("eval_interval", 0))
    events = {s for s in checkpoints if 0 <= s <= total_steps}
    if eval_interval > 0:
        events.update(range(eval_interval, total_steps + 1, eval_interval))
    events.add(total_steps)
    events = sorted(x for x in events if 0 <= x <= total_steps)

    ckpt_dir = out_dir / "backbone_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(bcfg["lr"]), weight_decay=float(bcfg.get("weight_decay", 0.0)))
    last_step = 0
    if 0 in checkpoints:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": 0, "config": cfg, "vocab": vocab.to_jsonable()}, ckpt_dir / "step_0.pt")
    # Initial eval.
    ev0 = eval_loss(model, val_ds, batches=int(bcfg.get("eval_batches", 20)), batch_size=int(bcfg["batch_size"]), device=device, precision=precision)
    write_eval_record(out_dir / "metrics_val.jsonl", {"step": 0, **ev0})

    for ev in events:
        if ev == 0:
            continue
        if ev > last_step:
            stats = train_steps(
                model,
                train_ds,
                steps=ev - last_step,
                batch_size=int(bcfg["batch_size"]),
                lr=float(bcfg["lr"]),
                weight_decay=float(bcfg.get("weight_decay", 0.0)),
                warmup_steps=int(bcfg.get("warmup_steps", 0)),
                device=device,
                precision=precision,
                log_interval=int(bcfg.get("log_interval", 50)),
                log_path=out_dir / "metrics_backbone_train.jsonl",
                optimizer=opt,
                global_step_offset=last_step,
            )
            last_step = ev
        if ev % max(1, eval_interval) == 0 or ev in checkpoints:
            evm = eval_loss(model, val_ds, batches=int(bcfg.get("eval_batches", 20)), batch_size=int(bcfg["batch_size"]), device=device, precision=precision)
            write_eval_record(out_dir / "metrics_val.jsonl", {"step": ev, **evm})
            print(f"EVAL step={ev} val_loss={evm['loss']:.4f} ppl={evm['ppl']:.2f}", flush=True)
        if ev in checkpoints:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": ev, "config": cfg, "vocab": vocab.to_jsonable()}, ckpt_dir / f"step_{ev}.pt")
            print(f"BACKBONE checkpoint step={ev}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    icfg = cfg.get("injection", {})
    if bool(icfg.get("enabled", True)):
        inj_ckpt = int(icfg.get("checkpoint", total_steps))
        state_path = ckpt_dir / f"step_{inj_ckpt}.pt"
        if not state_path.exists():
            raise RuntimeError(f"injection checkpoint not found: {state_path}")
        state = torch.load(state_path, map_location=device)
        arms = list(icfg.get("arms", ["wordhop", "nohop", "facts"]))
        for arm in arms:
            cell_id = f"step_{inj_ckpt}_{arm}"
            cell_dir = out_dir / "cells" / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_model = build_tiny_model(cfg, len(vocab.id_to_token)).to(device)
            cell_model.load_state_dict(state["model"])
            if arm == "wordhop":
                pre = score_structural(cell_model, wordhop_probe, vocab, device, max_probe)
            elif arm == "nohop":
                pre = score_structural(cell_model, nohop_probe, vocab, device, max_probe)
            elif arm == "facts":
                pre = score_facts(cell_model, facts_probe, vocab, device, max_probe)
            else:
                raise ValueError(f"unknown arm {arm}")
            dump_json(pre, cell_dir / "pre.json")
            inj_stats = train_steps(
                cell_model,
                arm_datasets[arm],
                steps=int(icfg["steps"]),
                batch_size=int(icfg["batch_size"]),
                lr=float(icfg["lr"]),
                weight_decay=float(icfg.get("weight_decay", 0.0)),
                warmup_steps=max(1, int(round(float(icfg.get("warmup_fraction", 0.05)) * int(icfg["steps"])))),
                device=device,
                precision=precision,
                log_interval=int(icfg.get("log_interval", 50)),
                log_path=cell_dir / "metrics_injection.jsonl",
            )
            if arm == "wordhop":
                post = score_structural(cell_model, wordhop_probe, vocab, device, max_probe)
            elif arm == "nohop":
                post = score_structural(cell_model, nohop_probe, vocab, device, max_probe)
            else:
                post = score_facts(cell_model, facts_probe, vocab, device, max_probe)
            dump_json(post, cell_dir / "uptake.json")
            retention = None
            if bool(cfg.get("washout", {}).get("enabled", True)):
                wcfg = cfg["washout"]
                train_steps(
                    cell_model,
                    train_ds,
                    steps=int(wcfg["steps"]),
                    batch_size=int(wcfg["batch_size"]),
                    lr=float(wcfg["lr"]),
                    weight_decay=float(wcfg.get("weight_decay", 0.0)),
                    warmup_steps=0,
                    device=device,
                    precision=precision,
                    log_interval=int(wcfg.get("log_interval", 30)),
                    log_path=cell_dir / "metrics_washout.jsonl",
                )
                if arm == "wordhop":
                    retention = score_structural(cell_model, wordhop_probe, vocab, device, max_probe)
                elif arm == "nohop":
                    retention = score_structural(cell_model, nohop_probe, vocab, device, max_probe)
                else:
                    retention = score_facts(cell_model, facts_probe, vocab, device, max_probe)
                dump_json(retention, cell_dir / "retention.json")
            row: dict[str, Any] = {"checkpoint_step": inj_ckpt, "arm": arm, "inj_elapsed_sec": inj_stats["elapsed_sec"], "inj_ema_loss": inj_stats["ema_loss"]}
            if arm == "facts":
                metric = "mean_target_logprob_per_token"
                row.update({"metric": metric, "pre": pre.get(metric), "post": post.get(metric), "uptake": post.get(metric, 0.0) - pre.get(metric, 0.0)})
                if retention is not None:
                    row.update({"retention": retention.get(metric), "retention_delta": retention.get(metric, 0.0) - pre.get(metric, 0.0)})
            else:
                metric = "accuracy"
                row.update({"metric": metric, "pre": pre.get(metric), "post": post.get(metric), "uptake": post.get(metric, 0.0) - pre.get(metric, 0.0)})
                if retention is not None:
                    row.update({"retention": retention.get(metric), "retention_delta": retention.get(metric, 0.0) - pre.get(metric, 0.0)})
                for k, v in post.items():
                    if k.startswith("acc_attractor_"):
                        row[f"post_{k}"] = v
            summary_rows.append(row)
            print(f"CELL {cell_id} {metric} pre={row['pre']:.4f} post={row['post']:.4f}", flush=True)

    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()})
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)

    manifest = {
        "run_id": run_id,
        "device": str(device),
        "precision": precision,
        "parameter_count": n_params,
        "vocab_size": len(vocab.id_to_token),
        "context_length": ctx,
        "corpus": {**corpus_meta, "used_fallback_sources": used_fallback, "train_docs": len(train_texts), "val_docs": len(val_texts)},
        "checkpoints": checkpoints,
        "injection_enabled": bool(icfg.get("enabled", True)),
        "injection_checkpoint": int(icfg.get("checkpoint", total_steps)) if icfg else None,
        "injection_arms": list(icfg.get("arms", [])) if icfg else [],
    }
    dump_json(manifest, out_dir / "manifest.json")

    report = [
        "# BabyLM debug milestone report",
        "",
        f"Run id: `{run_id}`",
        "",
        "This is a debug milestone for the BabyLM-scale path. It tests corpus loading, backbone training, checkpoint save/load, injection, scoring, and washout. It is not final thesis evidence.",
        "",
        "## Corpus",
        "",
        f"- matched files: {corpus_meta.get('matched_files', 0)}",
        f"- documents loaded from corpus: {corpus_meta.get('documents_loaded', 0)}",
        f"- used fallback injection-source stream: {used_fallback}",
        f"- train docs: {len(train_texts)}",
        f"- validation docs: {len(val_texts)}",
        "",
        "## Model",
        "",
        f"- parameters: {n_params:,}",
        f"- vocab size: {len(vocab.id_to_token):,}",
        f"- context length: {ctx}",
        "",
        "## Backbone checkpoints",
        "",
        "- " + ", ".join(str(x) for x in checkpoints),
        "",
    ]
    if summary_rows:
        report.extend(["## Injection summary", "", "| arm | metric | pre | post | uptake | retention |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in summary_rows:
            ret = row.get("retention")
            report.append(f"| {row['arm']} | {row['metric']} | {row.get('pre'):.4f} | {row.get('post'):.4f} | {row.get('uptake'):.4f} | {'' if ret is None else f'{ret:.4f}'} |")
    (out_dir / "babylm_debug_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
