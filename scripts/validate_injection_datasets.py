#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cplm.injection.io import read_jsonl
from cplm.injection.validation import validate_fact_records, validate_hop_divergence, validate_structural_pair, validate_wordhop_records, write_dataset_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate existing injection datasets.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing generated *.jsonl files")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    d = Path(args.dataset_dir)
    errors: list[str] = []
    summaries = {}
    loaded = {}
    for path in sorted(d.glob("*hop_*.jsonl")):
        records = read_jsonl(path)
        loaded[path.stem] = records
        errs, summary = validate_wordhop_records(records)
        summaries[path.stem] = summary
        errors.extend([f"{path.name}: {e}" for e in errs])
    for arm in ["nohop", "tokenhop", "wordhop"]:
        train_key = f"{arm}_train"
        probe_key = f"{arm}_probe"
        if train_key in loaded and probe_key in loaded:
            errs, summary = validate_structural_pair(loaded[train_key], loaded[probe_key], arm)
            summaries[f"{arm}_v4_gate"] = summary
            errors.extend([f"{arm}: {e}" for e in errs])
    if "wordhop_probe" in loaded and "tokenhop_probe" in loaded:
        errs, summary = validate_hop_divergence(loaded["wordhop_probe"], loaded["tokenhop_probe"])
        summaries["tokenhop_wordhop_divergence"] = summary
        errors.extend([f"tokenhop_wordhop_divergence: {e}" for e in errs])
    facts_train = d / "facts_train.jsonl"
    facts_probe = d / "facts_probe.jsonl"
    if facts_train.exists() and facts_probe.exists():
        errs, summary = validate_fact_records(read_jsonl(facts_train), read_jsonl(facts_probe))
        summaries["facts"] = summary
        errors.extend([f"facts: {e}" for e in errs])
    report = {"passed": not errors, "config": {"dataset_dir": str(d)}, "summaries": summaries, "examples": {}, "errors": errors}
    out = Path(args.report) if args.report else d.parent / "dataset_report_revalidated.md"
    write_dataset_report(report, out)
    print(f"Validation: {'PASS' if not errors else 'FAIL'}")
    print(f"Report: {out}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
