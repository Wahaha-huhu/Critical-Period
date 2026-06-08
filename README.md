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
