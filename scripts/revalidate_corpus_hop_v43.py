#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cplm.injection.validation import (
    validate_fact_records,
    validate_hop_divergence,
    validate_structural_pair,
    validate_wordhop_records,
    write_dataset_report,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Revalidate an existing corpus-HOP dataset with the v4.3 gate logic.")
    ap.add_argument("--dataset-dir", required=True, help="Directory containing *_train.jsonl and *_probe.jsonl files.")
    ap.add_argument("--report", default=None, help="Output markdown report path.")
    args = ap.parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_report = Path(args.report or dataset_dir.parent / "dataset_report_v4_3_revalidated.md")

    data = {p.stem: read_jsonl(p) for p in dataset_dir.glob("*.jsonl")}
    errors: list[str] = []
    summaries: dict = {"revalidation_mode": {"gate": "v4.3", "dataset_dir": str(dataset_dir)}}

    for key in sorted(k for k in data if k.endswith("_train") or k.endswith("_probe")):
        if key.startswith(("wordhop", "nohop", "tokenhop")):
            errs, summary = validate_wordhop_records(data[key])
            summaries[key] = summary
            errors.extend([f"{key}: {e}" for e in errs])

    for arm in ["nohop", "tokenhop", "wordhop"]:
        train_key = f"{arm}_train"
        probe_key = f"{arm}_probe"
        if train_key in data and probe_key in data:
            errs, summary = validate_structural_pair(data[train_key], data[probe_key], arm)
            summaries[f"{arm}_v4_3_gate"] = summary
            errors.extend([f"{arm}: {e}" for e in errs])

    if "wordhop_probe" in data and "tokenhop_probe" in data:
        errs, summary = validate_hop_divergence(data["wordhop_probe"], data["tokenhop_probe"])
        summaries["tokenhop_wordhop_divergence"] = summary
        errors.extend([f"tokenhop_wordhop_divergence: {e}" for e in errs])

    if "facts_train" in data and "facts_probe" in data:
        fact_errs, fact_summary = validate_fact_records(data["facts_train"], data["facts_probe"])
        summaries["facts"] = fact_summary
        errors.extend([f"facts: {e}" for e in fact_errs])

    examples = {
        "Corpus WORDHOP probes": data.get("wordhop_probe", [])[:5],
        "Corpus NOHOP probes": data.get("nohop_probe", [])[:5],
        "Factual probes": data.get("facts_probe", [])[:6],
    }
    report = {
        "passed": not errors,
        "config": {"report": str(out_report), "dataset_dir": str(dataset_dir), "gate": "v4.3"},
        "summaries": summaries,
        "examples": examples,
        "errors": errors,
    }
    write_dataset_report(report, out_report)
    print(f"Wrote v4.3 revalidation report to {out_report}")
    print(f"Validation: {'PASS' if not errors else 'FAIL'}")
    if errors:
        print(f"Errors: {len(errors)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
