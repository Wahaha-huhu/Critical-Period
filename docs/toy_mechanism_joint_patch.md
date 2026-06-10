# Toy mechanism joint-feature patch

The first toy run was useful as a pipeline check but not as a mechanism result. It used separate `T_t` and `C_c` tokens, and the class-decoding probe read hidden states at the class-token position. This made class decodability trivially perfect and prevented the run from testing whether an unused feature direction is discarded.

This patch adds a joint-feature representation:

```text
<bos> X_{type,class} Q_BASE  Y_T_type <eos>
<bos> X_{type,class} Q_CLASS Y_C_class <eos>
```

The base task requires only the type component of `X`; the injected task later requires the class component. Decodability is now measured at the query position, where the model makes the label decision, rather than at the raw class-token position. Injection defaults are also weaker to avoid immediate ceiling.

Use:

```bash
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_joint_smoke.yaml
```

Then run S1/S2:

```bash
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_joint_s1.yaml
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_joint_s2.yaml
```

Interpretation gates:

- base accuracy should reach ceiling;
- class pre-score should remain near chance before injection;
- query-position class decodability should decline under S1 if the unused direction is collapsed;
- injection uptake/retention should decline with the same stage if the mechanism is working;
- S2 should be interpreted at matched base competence, not merely matched steps.
