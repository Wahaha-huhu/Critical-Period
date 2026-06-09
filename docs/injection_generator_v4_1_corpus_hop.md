# Injection generator v4.1: corpus-derived HOP for LM tiers

This repo keeps the synthetic v4 structural set for toy/from-scratch and plumbing tests, but adds a separate corpus-derived HOP path for BabyLM/Pythia-tier experiments.

The motivation is that synthetic v4 passed shortcut gates but is not a good scientific carrier distribution for a naturally trained language model. For BabyLM/Pythia, the structural carrier sentences should be natural sentences from the model's distribution, then perturbed with NOHOP/WORDHOP.

## Build the corpus-derived set

Edit `configs/corpus_hop_babylm_v41.yaml` and point `corpus.globs` to local BabyLM text files:

```yaml
corpus:
  globs:
    - /path/to/babylm_10m/**/*.txt
  use_demo_if_no_corpus: false
```

Then run:

```bash
PYTHONPATH=src python scripts/build_corpus_hop_datasets.py \
  --config configs/corpus_hop_babylm_v41.yaml
```

The default config has `use_demo_if_no_corpus: true` so the script can be smoke-tested without BabyLM files. A demo-fallback pass only validates the plumbing and gates; it is not BabyLM evidence.

## Outputs

The builder writes:

```text
results/dataset_validation/corpus_hop_v4_1/
  config_resolved.yaml
  dataset_report.md
  example_sheet.md
  corpus_audit_sheet.md
  scoring_sanity_checks.csv
  datasets/
    wordhop_train.jsonl
    wordhop_probe.jsonl
    nohop_train.jsonl
    nohop_probe.jsonl
    facts_train.jsonl
    facts_probe.jsonl
```

`TOKENHOP` is optional and off by default for the LM-tier main path.

## Gates added in v4.1

The corpus-derived path reuses the v4 structural shortcut gates and adds:

- carrier naturalness proxy, to be supplemented by base-model loss for final BabyLM/Pythia runs;
- value-from-position baseline using verb index, source length, and construction/frame;
- value/verb-index and value/length decorrelation checks;
- source-disjoint train/probe carrier splits;
- single-qualifying-verb policy for the first interpretable LM-tier pilot;
- audit sheet with accepted and rejected examples.

## Important caveat

The current parser is a conservative heuristic. It is intentionally strict and keeps only sentences with one clear qualifying present-tense verb. For final BabyLM/Pythia runs, manually inspect `corpus_audit_sheet.md`; if many parses look wrong, switch to a stronger parser-backed extractor before spending training compute.
