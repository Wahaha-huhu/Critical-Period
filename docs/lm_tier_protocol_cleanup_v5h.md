# v5h LM-tier cleanup package

This patch packages the frozen v5h neutral-marker placement probe for later Pythia/BabyLM injection.

It does not change the accepted structural WORDHOP/NOHOP split. It adds the remaining protocol pieces identified after validation:

1. memorisation-only fact control;
2. reserved natural washout split;
3. Pythia `<HOP>` token handling protocol;
4. placement scoring floor and margin definition;
5. a 30-row manual audit file to complete before injection.

## Build the package

First ensure the v5h placement validation folder exists:

```bash
results/dataset_validation/placement_hop_v5h_babylm_simplewiki_scientific
```

Then run:

```bash
PYTHONPATH=src python scripts/prepare_lm_tier_protocol_package.py \
  --config configs/lm_tier_protocol_v5h.yaml
```

The output is:

```text
results/lm_tier_protocol/placement_hop_v5h_lm_tier/
```

## Important outputs

```text
datasets/wordhop_train.jsonl
datasets/wordhop_probe.jsonl
datasets/nohop_train.jsonl
datasets/nohop_probe.jsonl
datasets/facts_train.jsonl
datasets/facts_probe.jsonl
datasets/washout_text.jsonl
lm_tier_injection_protocol_v5h.md
lm_tier_package_report.md
manual_audit_30_rows_to_complete.csv
source_validation/
```

## Acceptance before injection

Before Pythia/BabyLM injection:

- `facts_probe.jsonl` must contain memorisation-only probes;
- `washout_text.jsonl` must be present and disjoint from all structural carriers;
- complete the first 30 rows of `manual_audit_30_rows_to_complete.csv`;
- keep the source validation report with the package;
- use the `<HOP>` token protocol in `lm_tier_injection_protocol_v5h.md`.
