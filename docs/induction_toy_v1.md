# Induction toy v1

This is the first induction-centred toy artifact.  It replaces the previous present-but-unused feature toy as the main causal mechanism target.

## Base task

Each batch is freshly generated.  A sequence contains key-value pairs from a shared symbol vocabulary, followed by a query key and the matching value:

```text
query mode:  <bos> k1 v1 ... kn vn <query> kq vq <eos>
repeat mode: <bos> k1 v1 ... kn vn kq vq ... <eos>
```

Only the query-key position is supervised.  The model must predict `vq`, the value that followed the previous occurrence of `kq`.  Since keys and values are resampled every batch, the task cannot be solved by memorising global key-value identities.

## Why this is the formative target

The task directly probes an induction / in-context copying mechanism: at the query key, a useful head should attend to the value token after the previous occurrence of that key.  The run therefore logs both behavioural associative recall and a per-head prefix-matching induction score.

## Current gate

The v1 gate is intentionally narrow:

1. A two-layer transformer should show a sharp rise in recall and induction score under S1.
2. Recall should be reported by query distance, so positional shortcuts are visible.
3. A one-layer control is provided but not required for the first smoke.

Schedule calibration, novel-format injection, deficit recovery, and Pythia induction probes come after this gate passes.


## Practical v1 calibration note

The code supports the stricter shared-symbol query format, but the first smoke configs use `role_mode: split` and `sequence_mode: repeat` because this makes the engineering gate easier to debug: value tokens come from the same synthetic symbol family but from a held-out half of the vocabulary, and the second block repeats queried keys without an explicit answer delimiter. Once this two-layer run shows a stable induction transition, the stricter shared-role query format can be re-enabled as the next calibration step.

## Commands

```bash
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_smoke_s1.yaml
PYTHONPATH=src python scripts/summarize_induction_toy_base.py results/induction_toy/induction_toy_smoke_s1
```

For the first substantive S1 pilot:

```bash
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_s1.yaml
PYTHONPATH=src python scripts/summarize_induction_toy_base.py results/induction_toy/induction_toy_s1_seed0
```
