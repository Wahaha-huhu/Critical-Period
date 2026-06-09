# v4 structural calibration diagnostics

These configs are intended to run before BabyLM-scale training if the ordinary v4 fast pilot does not learn NOHOP.

## Full-LM structural stress test

```bash
python scripts/build_injection_datasets.py --config configs/injection_datasets.yaml
python scripts/run_fast_injection_pilot.py --config configs/fast_structural_strong_v4.yaml
```

This uses the ordinary causal-LM loss on full structural sentences, but increases structural injection to 3000 steps at LR 1e-3 and disables washout. It tests whether the tiny model can learn NOHOP/WORDHOP when signal and optimization strength are much larger.

Pass target:

- NOHOP post accuracy above 0.70.
- WORDHOP post accuracy above 0.60 or clear positive margin/placement movement.

## Marker-focused diagnostic

```bash
python scripts/run_fast_injection_pilot.py --config configs/fast_marker_focused_diagnostic_v4.yaml
```

This masks the injection loss so structural examples train only the marker token. It is not the final experiment objective. It is a diagnostic to distinguish full-sentence loss dilution from a broken generator/scorer or unlearnable rule.

Expected:

- NOHOP should learn quickly.
- WORDHOP should learn if the counted-placement prefix and S/P target are usable.

Interpretation:

- If marker-focused succeeds but full-LM fails, the issue is signal dilution.
- If marker-focused also fails, inspect generator/scorer/model capacity before running BabyLM.
