#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from cplm.injection.placement_hop_v5 import (
    build_placement_hop_dataset,
    load_corpus_sentences,
    validate_dataset,
    write_audit_files,
    write_jsonl,
    write_report,
    config_hash,
)
from cplm.injection.facts import build_factual_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Build neutral-marker placement-only HOP v5 datasets.")
    ap.add_argument("--config", default="configs/placement_hop_v5_babylm.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out or cfg.get("output_dir", "results/dataset_validation/placement_hop_v5"))
    out_dir.mkdir(parents=True, exist_ok=True)
    data, meta = build_placement_hop_dataset(cfg)

    fact_cfg = cfg.get("facts", {})
    if bool(fact_cfg.get("enabled", True)):
        facts = build_factual_dataset(n_train=int(fact_cfg.get("n_train", 200)), n_probe=int(fact_cfg.get("n_probe", 50)), seed=int(cfg.get("seed", 0)) + 1009)
        # Keep only memorisation probes for the light specificity control if requested.
        if fact_cfg.get("probe_depth", "memorization") == "memorization":
            facts["facts_probe"] = [r for r in facts["facts_probe"] if r.get("probe_type") == "memorization"] or facts["facts_probe"][: int(fact_cfg.get("n_probe", 50))]
        data.update(facts)

    dataset_dir = out_dir / "datasets"
    written = {name: write_jsonl(rows, dataset_dir / f"{name}.jsonl") for name, rows in data.items()}
    # Reference naturalness sample comes from same corpus; if none, demo references are already in cache metadata only, so use source carriers.
    corpus_cfg = cfg.get("corpus", {})
    refs, _ = load_corpus_sentences(list(corpus_cfg.get("globs", []) or []), limit=int(cfg.get("naturalness", {}).get("reference_sentences", 5000)))
    if not refs:
        refs = [r["source_text"] for r in data["wordhop_train"] + data["wordhop_probe"]]
    errors, gates = validate_dataset(data, refs, float(cfg.get("naturalness", {}).get("proxy_threshold", 1.5)))
    report = {
        "passed": not errors,
        "config": {"config_path": args.config, "config_hash": config_hash(cfg), "output_dir": str(out_dir), "seed": cfg.get("seed", 0)},
        "summary": {**meta, "written": written},
        "gates": gates,
        "errors": errors,
        "examples": {"wordhop_probe": data.get("wordhop_probe", [])[:5], "nohop_probe": data.get("nohop_probe", [])[:5], "facts_probe": data.get("facts_probe", [])[:5]},
    }
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(report["summary"], indent=2), encoding="utf-8")
    write_report(report, out_dir / "dataset_report.md")
    write_audit_files(out_dir, data)
    print(f"Wrote placement-HOP v5 datasets to {out_dir}")
    print("Validation:", "PASS" if not errors else "FAIL")
    if errors:
        print(f"Errors: {len(errors)}. See {out_dir / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
