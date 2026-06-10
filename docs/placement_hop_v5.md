# Placement-HOP v5: neutral-marker corpus probe

This version implements the revised roadmap's language-model-tier structural probe. The structural readout is placement only: a single neutral marker is placed either immediately after a lemmatised present-tense verb (NOHOP) or four words after that verb, skipping punctuation (WORDHOP). There is no S/P value arm, no subject-number label, no attractor label, and no TOKENHOP by default.

## Why v5 exists

The v4.2/v4.3 corpus path was scientifically useful but slow and fragile because it tried to recover subject number and attractors. The revised roadmap measures only the novel operation: counted marker placement. That reduces parsing to verb identity and lemma, making the corpus gate faster and more auditable.

## CPU-speed changes

The builder caches extracted candidates in `cache/placement_hop_v5/`. Rerunning with the same corpus/parser config reuses the JSONL cache and skips spaCy parsing. The parser uses `nlp.pipe` with configurable `batch_size` and `n_process`.

Recommended CPU settings:

```yaml
parser:
  batch_size: 512
  n_process: 4   # or the number of allocated CPU cores
cache:
  reuse: true
```

On an A100 node this remains CPU-bound; the GPU is not used. Use the cache to avoid rerunning spaCy.

## Commands

Demo smoke test, no corpus required:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py --config configs/placement_hop_v5_demo.yaml
```

BabyLM/local corpus build:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py --config configs/placement_hop_v5_babylm.yaml
```

Zip a handoff:

```bash
zip -r placement_hop_v5_validation.zip results/dataset_validation/placement_hop_v5_babylm
```

## Gate

Hard gates:

- train/probe carrier disjointness
- mode-slot baseline fails
- length-anchored baseline fails
- frame+length baseline fails
- adversarial position-only baseline fails
- verb-relative oracle succeeds
- naturalness proxy passes
- manual verb/lemma audit is clean

Diagnostics:

- marker/length correlation
- distinct marker slots
- parser rejection counts

## Output files

- `datasets/wordhop_train.jsonl`
- `datasets/wordhop_probe.jsonl`
- `datasets/nohop_train.jsonl`
- `datasets/nohop_probe.jsonl`
- optional `datasets/facts_train.jsonl`, `datasets/facts_probe.jsonl`
- `dataset_report.md`
- `example_sheet.md`
- `verb_lemma_audit_sample.csv`
- `verb_lemma_audit_instructions.md`
- `manifest.json`
