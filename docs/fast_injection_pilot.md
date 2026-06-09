# Fast injection pilot

This pilot is a compute-bounded end-to-end test before the BabyLM backbone run. It uses the validated WORDHOP, NOHOP, and fictional-fact datasets, trains a tiny LM on a controlled English-like base stream, saves a few checkpoints, and then runs injection plus washout cells.

It is not the main BabyLM evidence. It checks the implementation path:

1. base training
2. checkpoint save and load
3. pre-injection scoring
4. fixed-dose injection
5. fixed washout on ordinary text
6. uptake and retention reporting

## Run

```bash
python scripts/build_injection_datasets.py --config configs/injection_datasets.yaml
python scripts/run_fast_injection_pilot.py --config configs/fast_injection_pilot.yaml
```

## Outputs

The script writes a timestamped directory under `results/fast_pilot/`:

```text
manifest.json
config_resolved.json
vocab.json
metrics_base_train.jsonl
base_checkpoints/
cells/<checkpoint>_<arm>/
  pre.json
  uptake.json
  retention.json
  metrics_injection.jsonl
  metrics_washout.jsonl
summary.csv
pilot_report.md
```

## How to read it

For WORDHOP and NOHOP, the main fast-pilot metric is marker forced-choice accuracy on held-out probes. For facts, the main fast-pilot metric is mean target-span log probability per token.

The pilot passes as an implementation test if all cells run, the pre/uptake/retention files are written, and the summary values are finite. It is only a difficulty signal, not thesis evidence.
