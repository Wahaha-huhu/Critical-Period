#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def first_transition(rows: List[Dict[str, Any]], recall_threshold: float, head_threshold: float) -> int | None:
    for r in rows:
        if float(r.get("recall_accuracy", 0.0)) >= recall_threshold and float(r.get("induction_head_max_score", 0.0)) >= head_threshold:
            return int(r.get("step", -1))
    return None


def flatten_run(run_dir: Path, recall_threshold: float, head_threshold: float) -> Dict[str, Any]:
    rows = load_jsonl(run_dir / "metrics_eval.jsonl")
    cfg = load_yaml(run_dir / "config_resolved.yaml")
    base_cfg = cfg.get("base_train", {}) if isinstance(cfg, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if not rows:
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": "missing_metrics_eval",
        }
    best = max(rows, key=lambda r: float(r.get("recall_accuracy", 0.0)))
    final = rows[-1]
    trans = first_transition(rows, recall_threshold, head_threshold)
    # A schedule is a candidate for later injection only if it has formed the
    # induction circuit and keeps high recall at the final checkpoint.
    final_recall = float(final.get("recall_accuracy", 0.0))
    final_head = float(final.get("induction_head_max_score", 0.0))
    competence_pass = final_recall >= recall_threshold and final_head >= head_threshold
    late_rows = rows[len(rows) // 2 :] if rows else []
    late_min_recall = min((float(r.get("recall_accuracy", 0.0)) for r in late_rows), default=float("nan"))
    return {
        "run_id": cfg.get("run_id", run_dir.name) if isinstance(cfg, dict) else run_dir.name,
        "run_dir": str(run_dir),
        "status": "ok",
        "schedule": base_cfg.get("schedule", ""),
        "peak_lr": base_cfg.get("peak_lr", ""),
        "min_lr": base_cfg.get("min_lr", ""),
        "constant_lr": base_cfg.get("constant_lr", ""),
        "max_lr": base_cfg.get("max_lr", ""),
        "cycle_steps": base_cfg.get("cycle_steps", ""),
        "total_steps": base_cfg.get("total_steps", ""),
        "warmup_steps": base_cfg.get("warmup_steps", ""),
        "n_symbols": data_cfg.get("n_symbols", ""),
        "n_pairs": data_cfg.get("n_pairs", ""),
        "n_layers": model_cfg.get("n_layers", ""),
        "d_model": model_cfg.get("d_model", ""),
        "eval_points": len(rows),
        "transition_step": "" if trans is None else trans,
        "best_step": best.get("step", ""),
        "best_recall_accuracy": float(best.get("recall_accuracy", 0.0)),
        "best_induction_score": float(best.get("induction_score", 0.0)),
        "best_induction_head_max_score": float(best.get("induction_head_max_score", 0.0)),
        "final_recall_accuracy": final_recall,
        "final_induction_score": float(final.get("induction_score", 0.0)),
        "final_induction_head_max_score": final_head,
        "final_loss_reduction_vs_uniform": float(final.get("loss_reduction_vs_uniform", 0.0)),
        "final_mean_stable_rank": final.get("mean_stable_rank", ""),
        "final_mean_effective_rank": final.get("mean_effective_rank", ""),
        "final_mean_spectral_norm": final.get("mean_spectral_norm", ""),
        "late_min_recall_accuracy": late_min_recall,
        "competence_pass": int(bool(competence_pass)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize induction-toy base schedule calibration runs.")
    ap.add_argument("--root", default="results/induction_toy", help="Directory containing run subdirectories.")
    ap.add_argument("--glob", default="induction_toy_copy_*", help="Run directory glob under --root.")
    ap.add_argument("--out-dir", default=None, help="Output directory. Defaults to --root/schedule_calibration_summary.")
    ap.add_argument("--recall-threshold", type=float, default=0.95)
    ap.add_argument("--head-threshold", type=float, default=0.50)
    args = ap.parse_args()

    root = Path(args.root)
    runs = sorted([p for p in root.glob(args.glob) if p.is_dir() and (p / "metrics_eval.jsonl").exists()])
    if not runs:
        raise SystemExit(f"No runs matched {root / args.glob}")
    rows = [flatten_run(p, args.recall_threshold, args.head_threshold) for p in runs]
    out_dir = Path(args.out_dir) if args.out_dir else root / "schedule_calibration_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "induction_schedule_calibration_summary.csv"
    fieldnames = list(rows[0].keys())
    # Preserve new keys if malformed rows appear.
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pass_rows = [r for r in rows if r.get("competence_pass") == 1]
    lines = [
        "# Induction toy schedule calibration summary",
        "",
        f"Root: `{root}`",
        f"Runs matched: {len(rows)}",
        f"Competence gate: final recall >= {args.recall_threshold} and final induction-head score >= {args.head_threshold}",
        f"Runs passing competence gate: {len(pass_rows)} / {len(rows)}",
        "",
        "## Runs",
        "",
        "| run | schedule | transition | final recall | final head score | mean stable rank | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {run} | {sched} | {trans} | {rec:.3f} | {head:.3f} | {rank} | {passed} |".format(
                run=r.get("run_id", ""),
                sched=r.get("schedule", ""),
                trans=r.get("transition_step", ""),
                rec=float(r.get("final_recall_accuracy", 0.0)),
                head=float(r.get("final_induction_head_max_score", 0.0)),
                rank=("" if r.get("final_mean_stable_rank", "") == "" else f"{float(r.get('final_mean_stable_rank')):.2f}"),
                passed=r.get("competence_pass", ""),
            )
        )
    lines += [
        "",
        "## Reading rule",
        "",
        "Only schedules that pass the competence gate should be used for the later injection/recruitability comparison.",
        "If a schedule fails to form induction or loses final recall, it is a failed base schedule, not evidence for a closed critical window.",
    ]
    md_path = out_dir / "induction_schedule_calibration_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
