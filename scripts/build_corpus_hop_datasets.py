#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cplm.config import load_yaml, save_yaml, config_hash
from cplm.injection.corpus_hop import build_corpus_hop_dataset, naturalness_summary, load_corpus_sentences, make_demo_corpus
from cplm.injection.facts import build_factual_dataset
from cplm.injection.io import write_jsonl
from cplm.injection.validation import validate_fact_records, validate_hop_divergence, validate_structural_pair, validate_wordhop_records, write_dataset_report


def _write_audit(out_dir: Path, records: dict[str, list[dict]], meta: dict) -> None:
    lines = ["# Corpus HOP v4.1 audit sheet", ""]
    lines.append("## Source and parser")
    lines.append("")
    for k in ["source_kind", "parser", "single_qualifying_verb_policy", "heldout_frame", "n_raw_sentences", "n_parse_valid_candidates"]:
        lines.append(f"- **{k}**: `{meta.get(k)}`")
    lines.append("")
    for name in ["wordhop_train", "wordhop_probe", "nohop_train", "nohop_probe"]:
        if name not in records:
            continue
        lines.append(f"## {name} examples")
        lines.append("")
        for r in records[name][:30]:
            lines.append(f"- source: `{r.get('source_text')}`")
            lines.append(f"  transformed: `{r.get('correct_text')}`")
            lines.append(f"  verb: `{r.get('verb_inflected')}`→`{r.get('verb_lemma')}`, subject `{r.get('head_noun')}` / `{r.get('subject_number')}`, marker `{r.get('marker')}`")
            lines.append(f"  marker_index `{r.get('marker_index')}`, verb_index `{r.get('verb_index_transformed')}`, attractors `{r.get('attractor_numbers')}`, frame `{r.get('frame_shape')}`, split `{r.get('frame_split_type')}`")
        lines.append("")
    lines.append("## Rejected sentence sample")
    lines.append("")
    for r in meta.get("rejection_examples", [])[:50]:
        lines.append(f"- `{r['reason']}`: {r['sentence']}")
    (out_dir / "corpus_audit_sheet.md").write_text("\n".join(lines), encoding="utf-8")


def _write_parser_audit_csv(out_dir: Path, records: dict[str, list[dict]]) -> None:
    """Write a manual parser-audit sheet for the corpus-HOP gate.

    The automated gates can show that shortcuts fail, but the language-model tier
    also needs a manual linguistic audit: the parser must locate a real verb,
    subject, subject number, and attractors. The manual_* columns are blank on
    purpose; fill them during audit and keep the completed CSV with the run.
    """
    rows = []
    # Prefer probe examples, then train examples; WORDHOP contains all structural metadata.
    pool = list(records.get("wordhop_probe", [])) + list(records.get("wordhop_train", []))
    for r in pool[:200]:
        meta = r.get("metadata") or {}
        rows.append({
            "source_id": r.get("source_id"),
            "split": r.get("split"),
            "source_text": r.get("source_text"),
            "wordhop_text": r.get("correct_text"),
            "verb_inflected": r.get("verb_inflected"),
            "verb_lemma": r.get("verb_lemma"),
            "verb_index": r.get("verb_index_transformed"),
            "head_noun": r.get("head_noun"),
            "head_noun_index": r.get("head_noun_index"),
            "subject_number": r.get("subject_number"),
            "marker": r.get("marker"),
            "marker_index": r.get("marker_index"),
            "template": r.get("template"),
            "frame_shape": r.get("frame_shape"),
            "attractor_indices": ";".join(map(str, r.get("attractor_indices") or [])),
            "attractor_numbers": ";".join(map(str, r.get("attractor_numbers") or [])),
            "parser": meta.get("parser"),
            "subject_dep": meta.get("subject_dep"),
            "subject_pos": meta.get("subject_pos"),
            "subject_tag": meta.get("subject_tag"),
            "verb_pos": meta.get("verb_pos"),
            "verb_tag": meta.get("verb_tag"),
            "manual_verb_correct": "",
            "manual_subject_correct": "",
            "manual_subject_number_correct": "",
            "manual_attractor_labels_correct": "",
            "manual_wordhop_slot_correct": "",
            "manual_fatal_error": "",
            "manual_notes": "",
        })
    if not rows:
        return
    fields = list(rows[0].keys())
    with (out_dir / "parser_audit_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    guidance = """# Parser audit instructions\n\nOpen `parser_audit_sample.csv` and fill the `manual_*` columns for a sample of corpus-derived WORDHOP examples. Suggested gate before BabyLM injection:\n\n- fatal parse errors below 2–3%\n- subject-number errors below 3–5%\n- WORDHOP slot errors approximately 0%\n\nThis audit is separate from the statistical shortcut gate. A dataset can pass the automated gate but still fail the linguistic audit if the parser labels are wrong.\n"""
    (out_dir / "parser_audit_instructions.md").write_text(guidance, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build corpus-derived HOP v4.1 datasets for BabyLM/Pythia tiers.")
    ap.add_argument("--config", default="configs/corpus_hop_babylm_v41.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out or cfg.get("output_dir", "results/dataset_validation/corpus_hop_v4_1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0))
    corpus_cfg = cfg.get("corpus", {})
    word_cfg = cfg.get("wordhop", {})
    fact_cfg = cfg.get("facts", {})

    parser_cfg = cfg.get("parser", {})
    wordhop, corpus_meta = build_corpus_hop_dataset(
        corpus_globs=list(corpus_cfg.get("globs", []) or []),
        n_train=int(word_cfg.get("n_train", 1000)),
        n_probe=int(word_cfg.get("n_probe", 300)),
        seed=seed,
        hop_distance=int(word_cfg.get("hop_distance", 4)),
        include_tokenhop=bool(word_cfg.get("include_tokenhop", False)),
        use_demo_if_no_corpus=bool(corpus_cfg.get("use_demo_if_no_corpus", True)),
        max_corpus_sentences=corpus_cfg.get("max_sentences"),
        parser_backend=str(parser_cfg.get("backend", "heuristic")),
        spacy_model=str(parser_cfg.get("spacy_model", "en_core_web_sm")),
        spacy_batch_size=int(parser_cfg.get("batch_size", 128)),
        require_parser=bool(parser_cfg.get("require_parser", False)),
    )
    facts = build_factual_dataset(
        n_train=int(fact_cfg.get("n_train", 200)),
        n_probe=int(fact_cfg.get("n_probe", 50)),
        seed=seed + 1009,
    )
    all_data = {**wordhop, **facts}
    dataset_dir = out_dir / "datasets"
    n_written = {name: write_jsonl(records, dataset_dir / f"{name}.jsonl") for name, records in all_data.items()}

    errors: list[str] = []
    summaries: dict = {"written": n_written, "corpus_source": corpus_meta}

    for key in sorted(k for k in wordhop if k.endswith("_train") or k.endswith("_probe")):
        errs, summary = validate_wordhop_records(wordhop[key])
        summaries[key] = summary
        errors.extend([f"{key}: {e}" for e in errs])

    for arm in ["nohop", "tokenhop", "wordhop"]:
        train_key = f"{arm}_train"
        probe_key = f"{arm}_probe"
        if train_key in wordhop and probe_key in wordhop:
            errs, summary = validate_structural_pair(wordhop[train_key], wordhop[probe_key], arm)
            summaries[f"{arm}_v4_1_gate"] = summary
            errors.extend([f"{arm}: {e}" for e in errs])
    if "wordhop_probe" in wordhop and "tokenhop_probe" in wordhop:
        errs, summary = validate_hop_divergence(wordhop["wordhop_probe"], wordhop["tokenhop_probe"])
        summaries["tokenhop_wordhop_divergence"] = summary
        errors.extend([f"tokenhop_wordhop_divergence: {e}" for e in errs])

    # Naturalness proxy on unmarked carriers. For final BabyLM/Pythia gates,
    # replace/supplement this with base-model per-token loss.
    carrier_texts = [r["source_text"] for r in wordhop.get("wordhop_train", []) + wordhop.get("wordhop_probe", [])]
    ref_globs = list(corpus_cfg.get("globs", []) or [])
    ref_texts = load_corpus_sentences(ref_globs, limit=5000)
    if not ref_texts:
        ref_texts = make_demo_corpus(max(2000, len(carrier_texts) * 2), seed=seed + 33)
    nat_errs, nat_summary = naturalness_summary(carrier_texts, ref_texts, float(cfg.get("naturalness", {}).get("proxy_threshold", 1.5)))
    summaries["naturalness_proxy"] = nat_summary
    errors.extend([f"naturalness: {e}" for e in nat_errs])

    fact_errs, fact_summary = validate_fact_records(facts["facts_train"], facts["facts_probe"])
    summaries["facts"] = fact_summary
    errors.extend([f"facts: {e}" for e in fact_errs])

    sanity_rows = [
        {"case": "marker_margin_positive", "correct": "S", "incorrect": "P", "correct_logprob": -0.1, "incorrect_logprob": -2.0, "expected_correct": True},
        {"case": "marker_margin_negative", "correct": "P", "incorrect": "S", "correct_logprob": -3.0, "incorrect_logprob": -0.2, "expected_correct": False},
    ]
    with (out_dir / "scoring_sanity_checks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
        w.writeheader(); w.writerows(sanity_rows)

    examples = {
        "Corpus WORDHOP probes": wordhop.get("wordhop_probe", [])[:5],
        "Corpus NOHOP probes": wordhop.get("nohop_probe", [])[:5],
        "Factual probes": facts.get("facts_probe", [])[:6],
    }
    report = {
        "passed": not errors,
        "config": {"config_path": args.config, "config_hash": config_hash(cfg), "seed": seed, "output_dir": str(out_dir)},
        "summaries": summaries,
        "examples": examples,
        "errors": errors,
    }
    save_yaml(cfg, out_dir / "config_resolved.yaml")
    write_dataset_report(report, out_dir / "dataset_report.md")
    _write_audit(out_dir, wordhop, corpus_meta)
    _write_parser_audit_csv(out_dir, wordhop)

    example_lines = ["# Corpus-derived injection dataset example sheet", ""]
    for title, recs in examples.items():
        example_lines.append(f"## {title}"); example_lines.append("")
        for rec in recs:
            if "correct_text" in rec:
                example_lines.append(f"- source: `{rec['source_text']}`")
                example_lines.append(f"  correct: `{rec['correct_text']}`")
                example_lines.append(f"  marker position: `{rec['marker_index']}`, marker: `{rec['marker']}`, arm: `{rec['arm']}`")
            elif "prompt" in rec:
                example_lines.append(f"- prompt: `{rec['prompt']}` → target `{rec['target']}` ({rec['depth']})")
        example_lines.append("")
    (out_dir / "example_sheet.md").write_text("\n".join(example_lines), encoding="utf-8")

    print(f"Wrote corpus-derived HOP datasets to {out_dir}")
    print(f"Validation: {'PASS' if not errors else 'FAIL'}")
    if errors:
        print(f"Errors: {len(errors)}. See {out_dir / 'dataset_report.md'}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
