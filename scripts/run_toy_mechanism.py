#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import yaml
import torch

from cplm.toy.training import train_base, run_injection_cell


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
    out_dir = Path(cfg.get("output_dir", "results/toy_mechanism")) / run_id
    if out_dir.exists() and cfg.get("overwrite", False):
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = {"run_id": run_id, "device": str(device), "config": args.config}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    train_base(cfg, out_dir, device)
    rows = []
    for ckpt_step in cfg.get("injection", {}).get("checkpoint_steps", []):
        ckpt = out_dir / "checkpoints" / f"step_{int(ckpt_step)}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        cell_dir = out_dir / "injection_cells" / f"step_{int(ckpt_step)}"
        rows.append(run_injection_cell(ckpt, cfg, cell_dir, device, checkpoint_step=int(ckpt_step)))
    if rows:
        with (out_dir / "toy_injection_grid.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    report_lines = [
        "# Toy mechanism run report",
        "",
        f"Run id: `{run_id}`",
        f"Device: `{device}`",
        "",
        "## Outputs",
        "",
        "- `metrics_base_train.jsonl`: base training loss and LR.",
        "- `metrics_base_eval.jsonl`: base/class pre-score, spectral markers, class decodability.",
        "- `toy_injection_grid.csv`: class-rule uptake and retention by checkpoint.",
        "- `checkpoints/`: base checkpoints with optimizer state.",
        "",
        "## Interpretation notes",
        "",
        "The base task depends on the type feature while the class feature is present but unused. The injection task requires the class feature. A closing window is supported when class decodability and injection uptake decline after spectral consolidation under S1, and are retained under S2 at matched base competence.",
    ]
    (out_dir / "toy_mechanism_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote toy mechanism run to {out_dir}")


if __name__ == "__main__":
    main()
