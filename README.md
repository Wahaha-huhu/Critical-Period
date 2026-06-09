# Critical-period LM experiments

Implementation scaffold for the Stage 0 synthetic agreement pilot.

The repository is config-driven and keeps results cleanly separated from code. Stage 0 currently runs only two seeds by default.

## Result layout

```text
results/
  raw/<stage>/<group_id>/<run_id>/
    config_resolved.yaml
    manifest.json
    metrics_train.jsonl
    metrics_eval.jsonl
    counts.jsonl
    checkpoints/
  summaries/<stage>/<group_id>/
    runs.csv
    latest_metrics.csv
    dynamics_train.csv
    dynamics_eval.csv
    dynamics_counts.csv
    gate_report.md
    plots/
      train_loss_ema.png
      learning_rate.png
      val_invariance.png
      val_acc_local.png
      val_acc_pp_same.png
      val_acc_pp_opp.png
      pp_opp_count.png
  logs/<stage>/
  manifests/<stage>/
  indices/runs.csv
```

Each run directory is self-contained. It stores the resolved config, manifest, streaming metrics, counts, and optional checkpoints. Summaries and plots are derived artifacts and can be regenerated from `results/raw`.

## Install

```bash
pip install -e .
```

If PyTorch with CUDA is not already installed, install the correct CUDA build first, then install the repo.

## Quick smoke test

```bash
python scripts/smoke_test.py --config configs/smoke.yaml
```

This runs a tiny CPU-friendly training loop, verifies generation, packing, evaluation, metric logging, and summary generation.

## Stage 0 pilot

```bash
python scripts/run_stage0_grid.py --config configs/stage0.yaml
python scripts/summarize_stage0.py --results results/raw/stage0/<group_id>
```

During a long run, export partial training dynamics with:

```bash
python scripts/summarize_stage0.py --results results/raw/stage0/<group_id> --allow-partial
```

The summary command writes CSV files and plots for training loss, validation probes, learning rate, and actual PP-opp dose.

## Core implementation choices

- Deterministic counter-based synthetic generator keyed by run seed, step, and sentence index.
- Packed word-level sequences with full next-token loss and padding ignored.
- Three probes: Local, PP-same, PP-opp.
- Headline structural score: attractor invariance, requiring correctness on same-number and opposite-number attractor versions of the same frame.
- Actual generated counts are logged: Local, PP-same, PP-opp, tokens, and LR-weighted PP-opp count.
- Training dynamics are logged every `log_interval` steps and validation probes every `eval_interval` steps.
- S1 holds the terminal learning rate after the calibrated budget `T` during fixed-dose continuation.
- S2 fixed-dose is the clean dose-and-rate-matched readout; S1 fixed-dose is the realistic low-rate readout.
- Stage 0 intentionally excludes Pythia, BabyLM, full onset grid, SVD tracking, and burst washout.

## Injection dataset milestone

The first implementation milestone builds the probe package for the revised critical-period programme:

- `WORDHOP` structural arm, four words after the lemmatised verb with punctuation skipped.
- `NOHOP` near-native within-rule control, marker immediately after the lemmatised verb.
- `TOKENHOP` optional pilot difficulty step.
- Fictional-fact control with memorisation, semantic rephrasing, and simple compositional probes.

Build and validate the dataset package:

```bash
python scripts/build_injection_datasets.py --config configs/injection_datasets.yaml
```

Outputs are written to `results/dataset_validation/injection_milestone/` by default:

```text
config_resolved.yaml
dataset_report.md
example_sheet.md
scoring_sanity_checks.csv
datasets/
  wordhop_train.jsonl
  wordhop_probe.jsonl
  nohop_train.jsonl
  nohop_probe.jsonl
  tokenhop_train.jsonl
  tokenhop_probe.jsonl
  facts_train.jsonl
  facts_probe.jsonl
```

Revalidate an existing generated package:

```bash
python scripts/validate_injection_datasets.py \
  --dataset-dir results/dataset_validation/injection_milestone/datasets
```

Do not start BabyLM training until the dataset report passes and the example sheet has been manually inspected.

### v4 fine structural dose sweep

After the coarse v4 dose sweep, run the fine transition sweep:

```bash
python scripts/run_structural_dose_sweep.py --config configs/fast_structural_fine_sweep_v4.yaml
```

See `docs/structural_fine_sweep_v4.md`.
