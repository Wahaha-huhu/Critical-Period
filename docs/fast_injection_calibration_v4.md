# Fast injection calibration v4

This calibration is the next step after the v4 end-to-end smoke run. It keeps the validated v4 generator and runner, but strengthens the tiny pilot so that structural learning can be measured before BabyLM-scale compute.

Run:

```bash
python scripts/build_injection_datasets.py --config configs/injection_datasets.yaml
python scripts/run_fast_injection_pilot.py --config configs/fast_injection_calibration_v4.yaml
```

Then zip the latest result:

```bash
LATEST=$(ls -td results/fast_pilot/* | head -1)
GROUP=$(basename "$LATEST")
zip -r "fast_calibration_v4_handoff_$GROUP.zip" "$LATEST"
```

## Defaults

- checkpoint: 600 only
- arms: WORDHOP, NOHOP, facts
- injection steps: 300
- injection LR: 3e-4
- washout steps: 120
- context length: 96
- max train records per arm: 2000
- max probe records per arm: 300

## Interpretation gate

This is still not thesis evidence. It is a calibration gate before BabyLM.

Expected pass pattern:

- NOHOP should learn clearly above chance, ideally >0.65.
- WORDHOP should either improve meaningfully above chance or show margin improvement.
- Facts should improve in mean target log probability.

If NOHOP does not learn, increase tiny-pilot capacity or injection steps before judging WORDHOP. If NOHOP learns but WORDHOP does not, keep TOKENHOP/WORDHOP as the BabyLM difficulty bracket.
