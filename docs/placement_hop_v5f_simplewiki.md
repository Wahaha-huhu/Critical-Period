# Placement-HOP v5f: SimpleWiki-only quality pass

This patch keeps the v5e neutral-marker placement-only design and changes the recommended corpus source for the LM-tier validation pass.

Why: the v5e Gutenberg+SimpleWiki run passed the placement shortcut gates, but the manual examples still contained archaic or dialogue-heavy Gutenberg fragments. The revised roadmap uses placement only, so the remaining quality requirement is a clean verb/lemma carrier set. SimpleWiki is the cleanest BabyLM source for that purpose.

## Run

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5f_babylm_simplewiki.yaml
```

Zip the output:

```bash
zip -r placement_hop_v5f_simplewiki_validation.zip \
  results/dataset_validation/placement_hop_v5f_babylm_simplewiki
```

If the build reports too few candidates, run the smaller fallback:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5f_babylm_simplewiki_small.yaml

zip -r placement_hop_v5f_simplewiki_small_validation.zip \
  results/dataset_validation/placement_hop_v5f_babylm_simplewiki_small
```

## CPU/runtime notes

This step is CPU-bound spaCy parsing. GPU type does not matter. The config uses `parser.max_candidates: 8000` and a reusable cache so the first run stops after enough clean candidates and later sampler-only reruns are much faster.

## Acceptance

Use the dataset only if both are true:

1. automated placement gate passes: mode-slot, length-anchored, frame+length, and adversarial position-only baselines fail while the verb-relative oracle succeeds;
2. audit quality is acceptable in `verb_lemma_audit_sample.csv` and `example_sheet.md`: target is a real verb, lemma is correct, and carrier sentences are not headings, transcript fragments, or dialogue noise.
