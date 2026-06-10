#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--recall-threshold", type=float, default=0.5)
    ap.add_argument("--induction-threshold", type=float, default=0.25)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = load_jsonl(run_dir / "metrics_eval.jsonl")
    if not rows:
        raise SystemExit("No eval rows found")
    best = max(rows, key=lambda r: r.get("recall_accuracy", 0.0))
    trans = None
    for r in rows:
        if r.get("recall_accuracy", 0.0) >= args.recall_threshold and r.get("induction_head_max_score", 0.0) >= args.induction_threshold:
            trans = r
            break
    lines = [
        "# Induction toy base summary",
        "",
        f"Run dir: `{run_dir}`",
        f"Eval points: {len(rows)}",
        "",
        "## Best recall point",
        "",
        f"step: {best.get('step')}",
        f"recall_accuracy: {best.get('recall_accuracy'):.4f}",
        f"recall_loss: {best.get('recall_loss'):.4f}",
        f"induction_score: {best.get('induction_score'):.4f}",
        f"induction_head_max_score: {best.get('induction_head_max_score'):.4f}",
        f"loss_reduction_vs_uniform: {best.get('loss_reduction_vs_uniform'):.4f}",
        "",
        "## Transition gate",
        "",
    ]
    if trans:
        lines += [
            "Status: PASS under the configured thresholds.",
            f"transition_step: {trans.get('step')}",
        ]
    else:
        lines += [
            "Status: NOT YET under the configured thresholds.",
            "Interpretation: use this as calibration evidence, not as the final induction formation result.",
        ]
    # Include distance profile at the final point.
    final = rows[-1]
    dist_keys = sorted([k for k in final if k.startswith("acc_dist_")], key=lambda x: int(x.split("_")[-1]))
    lines += ["", "## Final recall by query distance", ""]
    for k in dist_keys:
        lines.append(f"{k}: {final[k]:.4f}")
    out = run_dir / "induction_toy_base_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
