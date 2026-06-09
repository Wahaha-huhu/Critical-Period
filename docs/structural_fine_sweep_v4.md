# Structural fine dose sweep v4

This run refines the transition region found by the coarse v4 dose sweep. The coarse sweep showed that `lr=3e-4, steps=300` was underpowered, while `lr=3e-4, steps=600` made WORDHOP learn but pushed NOHOP close to ceiling. The fine sweep tests lower learning rates and intermediate step counts to choose a non-ceiling dose before BabyLM-scale pilots.

Run:

```bash
python scripts/build_injection_datasets.py --config configs/injection_datasets.yaml
python scripts/run_structural_dose_sweep.py --config configs/fast_structural_fine_sweep_v4.yaml
```

Then zip the newest output:

```bash
LATEST=$(ls -td results/fast_pilot/* | head -1)
GROUP=$(basename "$LATEST")
zip -r "fast_structural_fine_sweep_v4_handoff_$GROUP.zip" "$LATEST"
```

Acceptance target:

- NOHOP post-injection accuracy: 0.70 to 0.95.
- WORDHOP post-injection accuracy: 0.60 to 0.90.
- Prefer the smallest dose that lands in band, because the BabyLM model may learn faster than the tiny pilot model.

If no cell lands in band, inspect whether the sweep transitions sharply from underpowered to ceiling. If so, choose the lowest WORDHOP-learnable cell as a BabyLM bracket point rather than a final fixed dose.
