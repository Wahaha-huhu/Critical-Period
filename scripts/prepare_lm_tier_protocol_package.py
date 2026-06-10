#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from cplm.injection.facts import build_factual_dataset
from cplm.injection.placement_hop_v5 import load_corpus_sentences, stable_hash, normalize_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def source_hashes(rows: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for r in rows:
        text = r.get("source_text") or r.get("text") or ""
        if text:
            hashes.add(stable_hash(text))
    return hashes


def reserve_washout(cfg: dict[str, Any], structural_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wash_cfg = cfg.get("washout", {}) or {}
    if not wash_cfg.get("enabled", True):
        return [], {"enabled": False, "n": 0}
    globs = list(wash_cfg.get("globs", []) or [])
    sents, matched = load_corpus_sentences(globs, limit=None)
    if not matched:
        if wash_cfg.get("allow_if_corpus_missing", False):
            return [], {"enabled": True, "error": "no corpus matched", "matched_files": []}
        raise RuntimeError(f"washout enabled but no corpus files matched globs: {globs}")
    used = source_hashes(structural_rows)
    n = int(wash_cfg.get("n_sentences", 5000))
    min_words = int(wash_cfg.get("min_words", 8))
    max_words = int(wash_cfg.get("max_words", 80))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sent in sents:
        words = normalize_text(sent).split()
        if len(words) < min_words or len(words) > max_words:
            continue
        h = stable_hash(sent)
        if h in used or h in seen:
            continue
        seen.add(h)
        out.append({"id": f"washout_{len(out):05d}", "text": sent, "source_hash": h, "split": "washout"})
        if len(out) >= n:
            break
    meta = {
        "enabled": True,
        "matched_files": matched,
        "requested": n,
        "written": len(out),
        "min_words": min_words,
        "max_words": max_words,
        "disjoint_from_structural": len({r["source_hash"] for r in out} & used) == 0,
    }
    return out, meta


def make_audit_to_complete(audit_path: Path, out_path: Path, n: int) -> dict[str, Any]:
    if not audit_path.exists():
        return {"available": False, "reason": f"missing {audit_path}"}
    rows: list[dict[str, str]] = []
    with audit_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            row = dict(row)
            for col in [
                "manual_verb_correct",
                "manual_lemma_correct",
                "manual_carrier_clean",
                "manual_wordhop_slot_correct",
                "manual_fatal_error",
                "manual_notes",
            ]:
                row.setdefault(col, "")
            rows.append(row)
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {"available": True, "rows_to_complete": len(rows), "path": str(out_path)}


def write_protocol_summary(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# LM-tier package report")
    lines.append("")
    lines.append(f"Status: {'PASS' if manifest.get('passed') else 'CHECK'}")
    lines.append("")
    lines.append("## Structural probe")
    lines.append("")
    lines.append("Frozen structural source: `{}`".format(manifest["placement_dir"]))
    lines.append("")
    for k, v in manifest.get("structural_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Facts")
    lines.append("")
    for k, v in manifest.get("facts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Washout")
    lines.append("")
    for k, v in manifest.get("washout", {}).items():
        if k == "matched_files":
            lines.append(f"- matched_files: {len(v)}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Pythia token protocol")
    lines.append("")
    lines.append("- Add `<HOP>` as an additional special token.")
    lines.append("- Resize embeddings and LM head.")
    lines.append("- Initialise the new row with the mean existing embedding unless otherwise pre-registered.")
    lines.append("- Pre-score after resizing and before injection.")
    lines.append("- Log marker embedding update norm, marker embedding gradient norm, transformer block gradient norm, and embedding/deep-gradient ratio.")
    lines.append("")
    lines.append("## Remaining human action")
    lines.append("")
    lines.append("Complete the 30-row manual audit CSV before injection and archive it with this package.")
    (out_dir / "lm_tier_package_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare final LM-tier protocol package around frozen v5h placement probe.")
    ap.add_argument("--config", default="configs/lm_tier_protocol_v5h.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    placement_dir = Path(cfg["placement_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not placement_dir.exists():
        raise RuntimeError(f"placement_dir does not exist: {placement_dir}")
    src_ds = placement_dir / "datasets"
    if not src_ds.exists():
        raise RuntimeError(f"placement dataset folder missing: {src_ds}")
    dst_ds = out_dir / "datasets"
    dst_ds.mkdir(parents=True, exist_ok=True)

    structural_names = ["wordhop_train", "wordhop_probe", "nohop_train", "nohop_probe"]
    structural_rows: list[dict[str, Any]] = []
    structural_counts: dict[str, int] = {}
    for name in structural_names:
        rows = read_jsonl(src_ds / f"{name}.jsonl")
        structural_rows.extend(rows)
        structural_counts[name] = write_jsonl(rows, dst_ds / f"{name}.jsonl")

    facts_meta: dict[str, Any] = {"enabled": False}
    if cfg.get("facts", {}).get("enabled", True):
        fcfg = cfg["facts"]
        facts = build_factual_dataset(
            n_train=int(fcfg.get("train_entities", 200)),
            n_probe=int(fcfg.get("probe_entities", 50)),
            seed=int(cfg.get("seed", 17)) + 1009,
        )
        facts_probe = [r for r in facts["facts_probe"] if r.get("depth") == "memorization"]
        # Defensive cap in case the generator changes later.
        facts_probe = facts_probe[: int(fcfg.get("probe_entities", 50))]
        facts_train_n = write_jsonl(facts["facts_train"], dst_ds / "facts_train.jsonl")
        facts_probe_n = write_jsonl(facts_probe, dst_ds / "facts_probe.jsonl")
        facts_meta = {
            "enabled": True,
            "background_train_entities": int(fcfg.get("train_entities", 200)),
            "probe_entities": int(fcfg.get("probe_entities", 50)),
            "facts_train_records": facts_train_n,
            "facts_probe_records": facts_probe_n,
            "probe_depths": sorted(set(r.get("depth") for r in facts_probe)),
            "note": "facts_train includes injected statements for both background train facts and held-out probe facts; facts_probe is memorisation-only.",
        }

    washout_rows, washout_meta = reserve_washout(cfg, structural_rows)
    if washout_rows:
        write_jsonl(washout_rows, dst_ds / "washout_text.jsonl")

    # Copy validation artifacts.
    artifacts = [
        "dataset_report.md",
        "example_sheet.md",
        "verb_lemma_audit_sample.csv",
        "verb_lemma_audit_instructions.md",
        "config_resolved.yaml",
        "manifest.json",
    ]
    copied = []
    for a in artifacts:
        if copy_if_exists(placement_dir / a, out_dir / "source_validation" / a):
            copied.append(a)
    audit_meta = make_audit_to_complete(
        placement_dir / "verb_lemma_audit_sample.csv",
        out_dir / "manual_audit_30_rows_to_complete.csv",
        int(cfg.get("audit", {}).get("n_rows_to_complete", 30)),
    )

    # Copy protocol doc if present.
    proto_src = Path("docs/lm_tier_injection_protocol_v5h.md")
    if proto_src.exists():
        shutil.copy2(proto_src, out_dir / "lm_tier_injection_protocol_v5h.md")

    manifest = {
        "passed": bool(washout_meta.get("disjoint_from_structural", True)) and facts_meta.get("enabled", True),
        "placement_dir": str(placement_dir),
        "output_dir": str(out_dir),
        "structural_counts": structural_counts,
        "facts": facts_meta,
        "washout": washout_meta,
        "copied_validation_artifacts": copied,
        "audit": audit_meta,
        "scoring": cfg.get("scoring", {}),
        "pythia_token": cfg.get("pythia_token", {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_protocol_summary(out_dir, manifest)
    print(f"Wrote LM-tier protocol package to {out_dir}")
    print("Facts probe depth:", facts_meta.get("probe_depths"))
    print("Washout records:", washout_meta.get("written", 0))


if __name__ == "__main__":
    main()
