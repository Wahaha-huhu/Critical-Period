# BabyLM debug milestone v4

This milestone is the bridge between the tiny v4 injection calibration and the real BabyLM-scale backbone.

It tests the full engineering path:

1. load a local corpus, or fall back to the validated v4 source-text stream;
2. build a debug causal-LM vocabulary;
3. train a small GPT-style decoder backbone;
4. save and reload checkpoints with optimizer state;
5. run the calibrated v4 injection setting (`lr=3e-4`, `600` steps);
6. score WORDHOP, NOHOP, and facts before/after injection;
7. optionally run washout and score retention.

This is not final thesis evidence. It is a compute-safety gate before the BabyLM/A100 run.

## Build datasets

```bash
python scripts/build_injection_datasets.py --config configs/injection_datasets.yaml
```

## Smoke run

```bash
python scripts/run_babylm_debug_milestone.py \
  --config configs/babylm_debug_smoke_v4.yaml
```

## Debug run

```bash
python scripts/run_babylm_debug_milestone.py \
  --config configs/babylm_debug_backbone_v4.yaml
```

## Optional local BabyLM corpus

Edit the config:

```yaml
corpus:
  globs:
    - /path/to/babylm_10m/**/*.txt
```

If no file matches and `fallback_to_injection_sources: true`, the script uses the unmarked source text from the v4 injection generator. This fallback is useful for testing but should not be described as BabyLM training.

## Expected output

```text
results/babylm_debug/<run_id>/
  manifest.json
  config_resolved.json
  vocab.json
  metrics_backbone_train.jsonl
  metrics_val.jsonl
  backbone_checkpoints/
    step_0.pt
    step_500.pt
    step_1000.pt
    step_2000.pt
  cells/
    step_2000_wordhop/
    step_2000_nohop/
    step_2000_facts/
  summary.csv
  babylm_debug_report.md
```

## Handoff zip

```bash
LATEST=$(ls -td results/babylm_debug/* | head -1)
GROUP=$(basename "$LATEST")
zip -r "babylm_debug_v4_handoff_$GROUP.zip" "$LATEST" \
  -x "*/backbone_checkpoints/*.pt"
```

Do not include model checkpoints unless we need to debug checkpoint loading.
