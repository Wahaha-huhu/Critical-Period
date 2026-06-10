#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import yaml
import torch

from cplm.induction.train import train_induction_base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(int(cfg.get("torch_num_threads", 1)))
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = cfg.get("run_id", Path(args.config).stem)
    out_dir = Path(cfg.get("output_dir", "results/induction_toy")) / run_id
    if out_dir.exists() and cfg.get("overwrite", False):
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "device": str(device), "config": args.config}, indent=2), encoding="utf-8")
    train_induction_base(cfg, out_dir, device)
    report = [
        "# Induction toy base run",
        "",
        f"Run id: `{run_id}`",
        f"Device: `{device}`",
        "",
        "This run trains a small causal transformer on fresh associative-recall sequences.",
        "The supervised position is the query key, whose next token is the value paired with the previous occurrence of that key.",
        "",
        "Core files:",
        "",
        "- `metrics_train.jsonl`: training loss and learning rate.",
        "- `metrics_eval.jsonl`: recall accuracy, recall-by-distance, induction attention score, loss reduction, and spectral metrics.",
        "- `checkpoints/`: base checkpoints for later schedule/injection grids.",
        "",
        "The first scientific gate is that recall accuracy and induction score rise sharply under the two-layer S1 run.",
    ]
    (out_dir / "induction_toy_base_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote induction toy base run to {out_dir}")


if __name__ == "__main__":
    main()
