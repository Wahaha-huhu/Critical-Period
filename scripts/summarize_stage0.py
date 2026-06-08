#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def read_final(run_dir: Path) -> dict | None:
    path = run_dir / "final.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        final = json.load(f)
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    row = {
        "run_id": manifest["run_id"],
        "stage": manifest["stage"],
        "group_id": manifest["group_id"],
        "schedule": manifest["schedule"],
        "onset_fraction": manifest["onset_fraction"],
        "onset_step": manifest["onset_step"],
        "seed": manifest["seed"],
        "readout": manifest["readout"],
        "step": final["step"],
        "dose_matched": final["dose_matched"],
        "pp_opp_count": final["counts"]["pp_opp"],
        "local_count": final["counts"]["local"],
        "pp_same_count": final["counts"]["pp_same"],
        "tokens_seen": final["counts"]["tokens"],
        "lr_weighted_pp_opp": final["lr_weighted_pp_opp"],
    }
    for split in ["val", "test"]:
        for key, val in final[split].items():
            row[f"{split}_{key}"] = val
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 0 runs")
    parser.add_argument("--results", required=True, help="Path to results/raw/stage0/<group_id>")
    args = parser.parse_args()
    group_raw = Path(args.results).resolve()
    rows = []
    for run_dir in sorted(p for p in group_raw.iterdir() if p.is_dir()):
        row = read_final(run_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No final.json files found under {group_raw}")
    df = pd.DataFrame(rows).sort_values(["schedule", "seed", "onset_fraction", "readout"])
    repo_root = Path(__file__).resolve().parents[1]
    stage = str(df["stage"].iloc[0])
    group_id = str(df["group_id"].iloc[0])
    out_dir = repo_root / "results" / "summaries" / stage / group_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "runs.csv", index=False)

    latest = df[[
        "run_id", "schedule", "onset_fraction", "seed", "readout", "step", "dose_matched",
        "pp_opp_count", "lr_weighted_pp_opp", "test_acc_local", "test_acc_pp_same", "test_acc_pp_opp", "test_invariance"
    ]]
    latest.to_csv(out_dir / "latest_metrics.csv", index=False)

    lines = ["# Stage 0 gate report", ""]
    lines.append(f"Runs summarized: {len(df)}")
    lines.append("")
    lines.append("## Mean test metrics by condition")
    mean = df.groupby(["schedule", "onset_fraction", "readout"])[["test_acc_local", "test_acc_pp_same", "test_acc_pp_opp", "test_invariance", "pp_opp_count"]].mean().reset_index()
    lines.append(mean.to_markdown(index=False))
    lines.append("")
    lines.append("## Dose match check")
    dose = df[df["readout"].str.contains("fixed_dose", na=False)][["run_id", "schedule", "seed", "dose_matched", "pp_opp_count", "lr_weighted_pp_opp"]]
    lines.append(dose.to_markdown(index=False))
    lines.append("")
    lines.append("Interpretation reminder: S2 fixed-dose is the clean dose-and-rate-matched readout. S1 fixed-dose is the realistic low-terminal-rate readout.")
    with open(out_dir / "gate_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
