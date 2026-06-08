#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_final(run_dir: Path) -> dict | None:
    final = read_json(run_dir / "final.json")
    manifest = read_json(run_dir / "manifest.json")
    if final is None or manifest is None:
        return None
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
        "elapsed_sec": final.get("elapsed_sec"),
    }
    for split in ["val", "test"]:
        for key, val in final[split].items():
            row[f"{split}_{key}"] = val
    return row


def attach_manifest(rows: list[dict[str, Any]], manifest: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        r["kind"] = kind
        r["run_id"] = manifest["run_id"]
        r["stage"] = manifest["stage"]
        r["group_id"] = manifest["group_id"]
        r["schedule"] = manifest["schedule"]
        r["onset_fraction"] = manifest["onset_fraction"]
        r["onset_step"] = manifest["onset_step"]
        r["seed"] = manifest["seed"]
        r["readout"] = manifest["readout"]
        out.append(r)
    return out


def collect_dynamics(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        manifest = read_json(run_dir / "manifest.json")
        if manifest is None:
            continue
        train_rows.extend(attach_manifest(read_jsonl(run_dir / "metrics_train.jsonl"), manifest, "train"))
        eval_rows.extend(attach_manifest(read_jsonl(run_dir / "metrics_eval.jsonl"), manifest, "eval"))
        count_rows.extend(attach_manifest(read_jsonl(run_dir / "counts.jsonl"), manifest, "counts"))
    return pd.DataFrame(train_rows), pd.DataFrame(eval_rows), pd.DataFrame(count_rows)


def condition_label(row: pd.Series) -> str:
    return f"{row['schedule']} onset={row['onset_fraction']} seed={row['seed']} {row['readout']}"


def make_plots(out_dir: Path, train_df: pd.DataFrame, eval_df: pd.DataFrame, count_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not train_df.empty:
        plt.figure(figsize=(10, 6))
        for _, g in train_df.sort_values("step").groupby("run_id"):
            label = condition_label(g.iloc[0])
            y = g["ema_loss"] if "ema_loss" in g else g["loss"]
            plt.plot(g["step"], y, label=label, linewidth=1)
        plt.xlabel("step")
        plt.ylabel("training loss EMA")
        plt.title("Training loss dynamics")
        plt.legend(fontsize=6, loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir / "train_loss_ema.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 6))
        for _, g in train_df.sort_values("step").groupby("run_id"):
            label = condition_label(g.iloc[0])
            plt.plot(g["step"], g["lr"], label=label, linewidth=1)
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title("Learning-rate schedule")
        plt.legend(fontsize=6, loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir / "learning_rate.png", dpi=160)
        plt.close()

    if not eval_df.empty:
        val = eval_df[eval_df.get("split", "val") == "val"].copy() if "split" in eval_df else eval_df.copy()
        if not val.empty:
            plt.figure(figsize=(10, 6))
            for _, g in val.sort_values("step").groupby("run_id"):
                label = condition_label(g.iloc[0])
                plt.plot(g["step"], g["invariance"], label=label, linewidth=1)
            plt.xlabel("step")
            plt.ylabel("validation invariance")
            plt.title("Structural invariance dynamics")
            plt.ylim(-0.02, 1.02)
            plt.legend(fontsize=6, loc="best")
            plt.tight_layout()
            plt.savefig(plot_dir / "val_invariance.png", dpi=160)
            plt.close()

            for metric, ylabel, fname in [
                ("acc_local", "validation Local accuracy", "val_acc_local.png"),
                ("acc_pp_same", "validation PP-same accuracy", "val_acc_pp_same.png"),
                ("acc_pp_opp", "validation PP-opp accuracy", "val_acc_pp_opp.png"),
            ]:
                if metric not in val:
                    continue
                plt.figure(figsize=(10, 6))
                for _, g in val.sort_values("step").groupby("run_id"):
                    label = condition_label(g.iloc[0])
                    plt.plot(g["step"], g[metric], label=label, linewidth=1)
                plt.xlabel("step")
                plt.ylabel(ylabel)
                plt.title(ylabel)
                plt.ylim(-0.02, 1.02)
                plt.legend(fontsize=6, loc="best")
                plt.tight_layout()
                plt.savefig(plot_dir / fname, dpi=160)
                plt.close()

    if not count_df.empty:
        plt.figure(figsize=(10, 6))
        for _, g in count_df.sort_values("step").groupby("run_id"):
            label = condition_label(g.iloc[0])
            plt.plot(g["step"], g["pp_opp"], label=label, linewidth=1)
        plt.xlabel("step")
        plt.ylabel("actual PP-opp examples seen")
        plt.title("Actual hard-example dose")
        plt.legend(fontsize=6, loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir / "pp_opp_count.png", dpi=160)
        plt.close()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows yet._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 0 runs and export training-dynamics plots")
    parser.add_argument("--results", required=True, help="Path to results/raw/stage0/<group_id>")
    parser.add_argument("--allow-partial", action="store_true", help="Write dynamics even before every run has final.json")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    group_raw = Path(args.results).resolve()
    if not group_raw.exists():
        raise SystemExit(f"Missing results directory: {group_raw}")
    run_dirs = sorted(p for p in group_raw.iterdir() if p.is_dir())
    if not run_dirs:
        raise SystemExit(f"No run directories found under {group_raw}")

    final_rows = [r for r in (read_final(run_dir) for run_dir in run_dirs) if r is not None]
    first_manifest = read_json(run_dirs[0] / "manifest.json") or {}
    repo_root = Path(__file__).resolve().parents[1]
    stage = str(first_manifest.get("stage", "stage0"))
    group_id = str(first_manifest.get("group_id", group_raw.name))
    out_dir = repo_root / "results" / "summaries" / stage / group_id
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, eval_df, count_df = collect_dynamics(run_dirs)
    if not train_df.empty:
        train_df.sort_values(["schedule", "seed", "onset_fraction", "readout", "step"]).to_csv(out_dir / "dynamics_train.csv", index=False)
    if not eval_df.empty:
        eval_df.sort_values(["schedule", "seed", "onset_fraction", "readout", "split", "step"]).to_csv(out_dir / "dynamics_eval.csv", index=False)
    if not count_df.empty:
        count_df.sort_values(["schedule", "seed", "onset_fraction", "readout", "step"]).to_csv(out_dir / "dynamics_counts.csv", index=False)
    if not args.no_plots:
        make_plots(out_dir, train_df, eval_df, count_df)

    lines = ["# Stage 0 gate report", ""]
    lines.append(f"Run directories found: {len(run_dirs)}")
    lines.append(f"Completed runs with final.json: {len(final_rows)}")
    lines.append("")

    if final_rows:
        df = pd.DataFrame(final_rows).sort_values(["schedule", "seed", "onset_fraction", "readout"])
        df.to_csv(out_dir / "runs.csv", index=False)
        latest = df[[
            "run_id", "schedule", "onset_fraction", "seed", "readout", "step", "dose_matched",
            "pp_opp_count", "lr_weighted_pp_opp", "elapsed_sec",
            "test_acc_local", "test_acc_pp_same", "test_acc_pp_opp", "test_invariance"
        ]]
        latest.to_csv(out_dir / "latest_metrics.csv", index=False)
        lines.append("## Mean test metrics by completed condition")
        mean = df.groupby(["schedule", "onset_fraction", "readout"])[["test_acc_local", "test_acc_pp_same", "test_acc_pp_opp", "test_invariance", "pp_opp_count"]].mean().reset_index()
        lines.append(markdown_table(mean))
        lines.append("")
        lines.append("## Dose match check")
        dose = df[df["readout"].str.contains("fixed_dose", na=False)][["run_id", "schedule", "seed", "dose_matched", "pp_opp_count", "lr_weighted_pp_opp"]]
        lines.append(markdown_table(dose))
        lines.append("")
    elif not args.allow_partial:
        raise SystemExit(f"No final.json files found under {group_raw}. Re-run with --allow-partial to export dynamics from active runs.")

    if not train_df.empty:
        last_train = train_df.sort_values("step").groupby("run_id").tail(1)
        status_cols = ["run_id", "schedule", "onset_fraction", "seed", "readout", "step", "loss", "ema_loss", "lr", "count_pp_opp", "elapsed_sec"]
        status_cols = [c for c in status_cols if c in last_train]
        lines.append("## Latest training status")
        lines.append(markdown_table(last_train[status_cols].sort_values("run_id")))
        lines.append("")

    lines.append("## Interpretation reminder")
    lines.append("S2 fixed-dose is the clean dose-and-rate-matched readout. S1 fixed-dose is the realistic low-terminal-rate readout.")
    lines.append("")
    lines.append("Training dynamics are exported to `dynamics_train.csv`, `dynamics_eval.csv`, `dynamics_counts.csv`, and `plots/`.")
    with open(out_dir / "gate_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
