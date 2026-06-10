# Induction toy v3: base schedule calibration

This patch moves the induction toy from "does the copy-repeat circuit form under S1?" to the next gate: **which schedules are valid matched-competence bases for the later injection/recruitability experiment?**

The patch deliberately does **not** add injection yet. Injection is only meaningful after each compared schedule has formed the induction circuit to comparable competence.

## Base task

The current base task is the copy-repeat induction task:

```text
<bos> s1 s2 ... sn  s1 s2 ... sn <eos>
```

At the second occurrence of each `si`, the model must predict `s{i+1}`. This implements the canonical induction pattern:

```text
[A][B] ... [A] -> [B]
```

The segment is freshly sampled each batch, so memorising fixed sequences is not a solution.

## New checkpoint discipline

The v2 result showed the transition occurs between step 100 and step 200. The old checkpoints skipped the transition itself. The dense configs now save and evaluate:

```text
[0, 50, 100, 150, 200, 250, 300, 500, 800, 1200, 1800, 2500, 3500, 5000]
```

This is the grid that later injection should inherit.

## Schedule families

### S1: warmup + cosine decay

```yaml
schedule: s1_decay
peak_lr: 7e-4
min_lr: 1e-5
warmup_steps: 300
```

### S2: warmup + constant LR

Candidate constant rates:

```text
1e-4, 3e-4, 5e-5
```

Only values that reach matched induction competence should be kept for injection.

### S3: warmup + cosine restarts

The initial restart candidate is:

```yaml
schedule: s3_cyclic
max_lr: 3e-4
min_lr: 3e-5
cycle_steps: 800
```

This is the cyclic/restart counterfactual. It is not a claim yet; it must pass the same competence gate as S1 and S2.

## Gate for later injection

A schedule is eligible for injection only if final evaluation satisfies:

```text
recall_accuracy >= 0.95
induction_head_max_score >= 0.50
```

The transition step is the first evaluated step passing the same thresholds.

## Run commands

```bash
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_copy_s1_dense.yaml
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_copy_s2_const1e4.yaml
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_copy_s2_const3e4.yaml
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_copy_s3_restart.yaml
```

Optional lower-rate constant schedule:

```bash
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_copy_s2_const5e5.yaml
```

One-layer diagnostic:

```bash
PYTHONPATH=src python scripts/run_induction_toy_base.py --config configs/induction_toy_copy_one_layer_control.yaml
```

Summarise:

```bash
PYTHONPATH=src python scripts/summarize_induction_schedule_calibration.py \
  --root results/induction_toy \
  --glob 'induction_toy_copy_*'
```

Zip handoff without checkpoints:

```bash
zip -r induction_toy_schedule_calibration_v3_handoff.zip results/induction_toy \
  -x "*/checkpoints/*.pt"
```

## Reading rule

Do not interpret a failed S2 or S3 as a critical-period result. If a schedule does not form induction, it is not a matched-competence counterfactual. The later plasticity question should compare only schedules that have already passed the base induction gate.
