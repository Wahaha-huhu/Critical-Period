# Structural dose sweep v4

This diagnostic chooses a non-ceiling injection dose before BabyLM-scale runs. It trains the tiny base LM once, saves the step-600 checkpoint, then reuses that checkpoint for each structural arm/dose cell.

Run:

```bash
python scripts/build_injection_datasets.py --config configs/injection_datasets.yaml
python scripts/run_structural_dose_sweep.py --config configs/fast_structural_dose_sweep_v4.yaml
```

Zip the latest output:

```bash
LATEST=$(ls -td results/fast_pilot/* | head -1)
GROUP=$(basename "$LATEST")
zip -r "fast_structural_dose_sweep_v4_handoff_$GROUP.zip" "$LATEST"
```

Interpretation gate:

- NOHOP should learn above 0.70 but should not be completely saturated for the chosen BabyLM pilot dose.
- WORDHOP should learn above 0.60 and ideally below 0.90.
- If all 1e-3 cells saturate, use a weaker dose or LR for BabyLM.
- If 3e-4 cells fail but 1e-3 cells saturate, add an intermediate LR such as 6e-4 in the next sweep.

This is still a tiny-model calibration, not final thesis evidence.
