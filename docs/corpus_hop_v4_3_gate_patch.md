# Corpus-HOP v4.3 gate patch

This patch implements the design revision after the v4.2 parser-backed corpus-HOP runs.

## Motivation

The v4.2 corpus-derived set had mostly passed the shortcut checks, but the remaining failures sometimes came from proxy correlations such as tail length after the marker versus verb position. Those correlations are useful diagnostics, but they are not themselves evidence that the marker can be placed without finding the verb. The decisive question is adversarial: can a predictor that never sees the verb or parse path place the marker?

## Gate change

v4.3 keeps the original C1 simple placement baselines as hard gates:

- global mode slot
- length-anchored slot
- frame plus length-bucket mode
- verb-relative oracle as a sanity check

It also adds a stronger adversarial position-only baseline. This predictor may use only features that are available without locating the verb:

- source sentence length
- construction/frame label
- punctuation counts
- punctuation-position signatures
- length buckets

It may not use:

- verb index
- subject number
- dependency path
- true marker slot
- tail length after the marker
- any feature computed from the true marker location

The adversarial baseline is now a hard gate: exact marker-slot accuracy must be below 0.25 on the probe split.

## Diagnostic-only fields

The following fields remain printed in the report but no longer reject the dataset by themselves:

- C2 verb-index spread
- C3 marker/length correlation
- C4 tail/verb correlation

If the adversarial position-only baseline fails, these proxy correlations are not actionable shortcuts. If it succeeds, the dataset is still invalid and must be resampled or redesigned.

## Parser audit

v4.3 also writes:

- `parser_audit_sample.csv`
- `parser_audit_instructions.md`

The CSV contains 200 corpus-derived WORDHOP examples with blank manual-audit columns. The automated shortcut gate is not enough for BabyLM/Pythia-tier claims; the parser must also be linguistically reliable.

Suggested manual gate:

- fatal parse errors below 2–3%
- subject-number errors below 3–5%
- WORDHOP slot errors approximately 0%

## Interpretation

A v4.3 PASS means the corpus-HOP dataset has passed the statistical shortcut gate. It does not yet mean the full BabyLM experiment is ready. The next gates are parser-audit completion, BabyLM-specific dose calibration, washout calibration, matched-competence S1/S2 pairing, and consolidation-spanning checkpoint selection.

## Revalidating without reparsing

If a v4.2/v4.2b/v4.2c run already produced JSONL datasets, you do not need to rerun spaCy parsing just to apply the revised v4.3 gate. Run:

```bash
PYTHONPATH=src python scripts/revalidate_corpus_hop_v43.py \
  --dataset-dir results/dataset_validation/corpus_hop_v4_2_spacy/datasets \
  --report results/dataset_validation/corpus_hop_v4_2_spacy/dataset_report_v4_3_revalidated.md
```

This reuses the existing generated records, adds the adversarial position-only baseline, and treats C2/C3/C4 as diagnostics.
