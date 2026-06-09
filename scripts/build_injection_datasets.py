#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cplm.config import load_yaml, save_yaml, config_hash
from cplm.injection.facts import build_factual_dataset
from cplm.injection.io import write_jsonl
from cplm.injection.validation import validate_disjoint_texts, validate_fact_records, validate_wordhop_records, write_dataset_report
from cplm.injection.wordhop import build_wordhop_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WORDHOP/NOHOP/TOKENHOP and factual injection datasets.")
    parser.add_argument("--config", default="configs/injection_datasets.yaml")
    parser.add_argument("--out", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out or cfg.get("output_dir", "results/dataset_validation/injection_milestone"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    word_cfg = cfg.get("wordhop", {})
    fact_cfg = cfg.get("facts", {})

    wordhop = build_wordhop_dataset(
        n_train=int(word_cfg.get("n_train", 2000)),
        n_probe=int(word_cfg.get("n_probe", 500)),
        seed=seed,
        hop_distance=int(word_cfg.get("hop_distance", 4)),
        include_tokenhop=bool(word_cfg.get("include_tokenhop", True)),
    )
    facts = build_factual_dataset(
        n_train=int(fact_cfg.get("n_train", 200)),
        n_probe=int(fact_cfg.get("n_probe", 50)),
        seed=seed + 1009,
    )

    dataset_dir = out_dir / "datasets"
    n_written: dict[str, int] = {}
    for name, records in {**wordhop, **facts}.items():
        n_written[name] = write_jsonl(records, dataset_dir / f"{name}.jsonl")

    # Validation over each generated file.
    errors: list[str] = []
    summaries = {"written": n_written}
    for key in ["nohop_train", "nohop_probe", "tokenhop_train", "tokenhop_probe", "wordhop_train", "wordhop_probe"]:
        if key in wordhop:
            errs, summary = validate_wordhop_records(wordhop[key])
            summaries[key] = summary
            errors.extend([f"{key}: {e}" for e in errs])
    # Structural probes must be held out from structural injection training.
    for arm in ["nohop", "tokenhop", "wordhop"]:
        train_key = f"{arm}_train"
        probe_key = f"{arm}_probe"
        if train_key in wordhop and probe_key in wordhop:
            split_errors, split_summary = validate_disjoint_texts(wordhop[train_key], wordhop[probe_key], arm)
            summaries[f"{arm}_split"] = split_summary
            errors.extend([f"{arm}: {e}" for e in split_errors])

    fact_errors, fact_summary = validate_fact_records(facts["facts_train"], facts["facts_probe"])
    summaries["facts"] = fact_summary
    errors.extend([f"facts: {e}" for e in fact_errors])

    # Minimal scoring sanity fixtures, designed to catch sign mistakes in later model adapters.
    sanity_rows = [
        {"case": "marker_margin_positive", "correct": "S", "incorrect": "P", "correct_logprob": -0.1, "incorrect_logprob": -2.0, "expected_correct": True},
        {"case": "marker_margin_negative", "correct": "P", "incorrect": "S", "correct_logprob": -3.0, "incorrect_logprob": -0.2, "expected_correct": False},
    ]
    sanity_path = out_dir / "scoring_sanity_checks.csv"
    with sanity_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sanity_rows)

    examples = {
        "WORDHOP probes": wordhop.get("wordhop_probe", [])[:3],
        "NOHOP probes": wordhop.get("nohop_probe", [])[:3],
        "Factual probes": facts.get("facts_probe", [])[:6],
    }
    report = {
        "passed": not errors,
        "config": {
            "config_path": args.config,
            "config_hash": config_hash(cfg),
            "seed": seed,
            "output_dir": str(out_dir),
        },
        "summaries": summaries,
        "examples": examples,
        "errors": errors,
    }
    save_yaml(cfg, out_dir / "config_resolved.yaml")
    write_dataset_report(report, out_dir / "dataset_report.md")

    # Compact example sheet for quick human inspection.
    example_lines = ["# Injection dataset example sheet", ""]
    for title, recs in examples.items():
        example_lines.append(f"## {title}")
        example_lines.append("")
        for rec in recs:
            if "correct_text" in rec:
                example_lines.append(f"- source: `{rec['source_text']}`")
                example_lines.append(f"  correct: `{rec['correct_text']}`")
                example_lines.append(f"  incorrect: `{rec['incorrect_text']}`")
                example_lines.append(f"  marker position: `{rec['marker_index']}`, marker: `{rec['marker']}`, arm: `{rec['arm']}`")
            elif "prompt" in rec:
                example_lines.append(f"- prompt: `{rec['prompt']}` → target `{rec['target']}` ({rec['depth']})")
            elif "text" in rec:
                example_lines.append(f"- `{rec['text']}`")
        example_lines.append("")
    (out_dir / "example_sheet.md").write_text("\n".join(example_lines), encoding="utf-8")

    print(f"Wrote injection datasets to {out_dir}")
    print(f"Validation: {'PASS' if not errors else 'FAIL'}")
    if errors:
        print(f"Errors: {len(errors)}. See {out_dir / 'dataset_report.md'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
