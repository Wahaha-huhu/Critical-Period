#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cplm.config import RunSpec, load_yaml, round_onset
from cplm.results import ResultManager
from cplm.train import train_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 0 two-by-two synthetic agreement pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = load_yaml(args.config)
    stage = str(config.get("stage", "stage0"))
    group_id = config.get("experiment", {}).get("group_id")
    if not group_id:
        group_id = f"{config['experiment']['name']}__{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config.setdefault("experiment", {})["group_id"] = group_id
    results_root = repo_root / str(config.get("results_root", "results"))

    total_steps = int(config["train"]["total_steps"])
    rounding = str(config["data"].get("onset_rounding", "nearest"))
    seeds = [int(s) for s in config["stage0"]["seeds"]]
    onsets = [float(x) for x in config["stage0"]["onsets"]]
    schedules = [str(x) for x in config["stage0"]["schedules"]]
    if len(onsets) != 2:
        raise ValueError("Stage 0 currently expects exactly two onsets, early and late")
    early_onset = min(onsets)
    late_onset = max(onsets)

    print(f"group_id={group_id}")
    print(f"stage={stage} seeds={seeds} onsets={onsets} schedules={schedules}")

    early_targets: dict[tuple[str, int], int] = {}
    fixed_age_finals: dict[str, dict] = {}

    for seed in seeds:
        for schedule in schedules:
            for onset in [early_onset, late_onset]:
                onset_step = round_onset(total_steps, onset, rounding)
                spec = RunSpec(stage=stage, group_id=group_id, schedule=schedule, onset_fraction=onset, onset_step=onset_step, seed=seed, readout="fixed_age")
                run_config = copy.deepcopy(config)
                run_config["run"] = spec.__dict__ | {"run_id": spec.run_id}
                print(f"RUN {spec.run_id}")
                if args.dry_run:
                    continue
                rm = ResultManager(repo_root, results_root, spec, run_config)
                final = train_run(run_config, rm, schedule=schedule, seed=seed, onset_step=onset_step, overwrite=args.overwrite)
                fixed_age_finals[spec.run_id] = final
                if onset == early_onset:
                    early_targets[(schedule, seed)] = int(final["counts"]["pp_opp"])

    if bool(config.get("stage0", {}).get("fixed_dose", True)) and not args.dry_run:
        for seed in seeds:
            for schedule in schedules:
                target = early_targets[(schedule, seed)]
                onset_step = round_onset(total_steps, late_onset, rounding)
                spec = RunSpec(
                    stage=stage,
                    group_id=group_id,
                    schedule=schedule,
                    onset_fraction=late_onset,
                    onset_step=onset_step,
                    seed=seed,
                    readout="fixed_dose_clean" if schedule == "s2_constant" else "fixed_dose_realistic",
                )
                run_config = copy.deepcopy(config)
                run_config["run"] = spec.__dict__ | {"run_id": spec.run_id, "target_pp_opp_count": target}
                # Conservative hard cap avoids infinite loops if a config is broken.
                max_steps = int(total_steps * 3)
                print(f"RUN {spec.run_id} target_pp_opp={target}")
                rm = ResultManager(repo_root, results_root, spec, run_config)
                train_run(
                    run_config,
                    rm,
                    schedule=schedule,
                    seed=seed,
                    onset_step=onset_step,
                    max_steps=max_steps,
                    target_pp_opp_count=target,
                    overwrite=args.overwrite,
                )

    manifest = {
        "group_id": group_id,
        "stage": stage,
        "config": str(Path(args.config).resolve()),
        "early_onset": early_onset,
        "late_onset": late_onset,
        "early_targets": {f"{k[0]}__seed-{k[1]}": v for k, v in early_targets.items()},
    }
    out = results_root / "summaries" / stage / group_id
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stage0_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"summary_dir={out}")


if __name__ == "__main__":
    main()
