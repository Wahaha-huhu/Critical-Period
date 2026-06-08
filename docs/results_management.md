# Results management contract

All experiments are written under `results/` and every run is self-contained.

## Raw run directory

```text
results/raw/<stage>/<group_id>/<run_id>/
  config_resolved.yaml
  manifest.json
  vocab.jsonl
  metrics_train.jsonl
  metrics_eval.jsonl
  counts.jsonl
  final.json
  checkpoints/
```

`config_resolved.yaml` is the exact config used for the run. `manifest.json` records the stage, schedule, onset, seed, readout, config hash, host, Python version, and git SHA when available.

## Derived summaries

```text
results/summaries/<stage>/<group_id>/
  stage0_manifest.json
  runs.csv
  latest_metrics.csv
  gate_report.md
```

Summaries are derived from raw runs and can be regenerated with:

```bash
python scripts/summarize_stage0.py --results results/raw/stage0/<group_id>
```

## Counts and dose matching

The fixed-dose rule uses actual generated PP-opp counts, not an analytic expectation. Each run logs:

- `local`
- `pp_same`
- `pp_opp`
- `sentences`
- `tokens`
- `lr_weighted_pp_opp`

For Stage 0, `s2_constant` fixed-dose is labelled `fixed_dose_clean`, because dose and rate are matched. `s1_decay` fixed-dose is labelled `fixed_dose_realistic`, because the matched dose after `T` is delivered at the held terminal learning rate.

## Version-control policy

Generated results are ignored by `.gitignore` by default. To archive a result group, copy the raw group and summary group to long-term storage with the same `<group_id>`.
