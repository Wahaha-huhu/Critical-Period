# Toy joint-feature schedule calibration patch

The first joint-feature toy run was an engineering pass, but not yet the intended schedule-causal result.

Observed issue:

- S1 shows a late weakening of class-rule uptake.
- S2 with `constant_lr=1e-3` weakens even more after late checkpoints.
- This means the S2 counterfactual was too aggressive: a high constant LR plus AdamW weight decay can itself keep shrinking and reshaping weights, causing stronger collapse rather than keeping the unused feature plastic.

This patch adds lower constant-rate S2 variants:

- `configs/toy_mechanism_joint_s2_const1e4.yaml`
- `configs/toy_mechanism_joint_s2_const3e4.yaml`
- `configs/toy_mechanism_joint_s2_const5e5.yaml`

It also adds deterministic geometry diagnostics to the decodability report:

- `class_pair_distance`: mean distance between class variants within each type at the base decision position.
- `type_pair_distance`: mean distance between type variants at the same position.
- `class_to_type_distance_ratio`: the class distance divided by type distance.

These are meant to complement the linear probe, which can be unstable in the tiny four-condition setting.

Recommended next run:

```bash
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_joint_s1.yaml
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_joint_s2_const3e4.yaml
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_joint_s2_const1e4.yaml
```

Then zip outputs without checkpoints:

```bash
zip -r toy_mechanism_joint_schedule_calibration_handoff.zip results/toy_mechanism \
  -x "*/checkpoints/*.pt"
```

Interpretation gate:

- Base accuracy/loss must be comparable across S1 and S2.
- S1 should show spectral consolidation and lower late class-rule uptake/retention.
- At least one lower-rate S2 should retain higher class geometry and higher late uptake at matched base competence.
- If no S2 variant keeps plasticity open, the toy still shows a window, but the schedule-cause claim needs a different counterfactual or a revised hypothesis about constant-LR plus weight decay.
