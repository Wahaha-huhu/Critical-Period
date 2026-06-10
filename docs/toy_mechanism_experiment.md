# Toy mechanism experiment

This experiment is the lead causal test in the revised roadmap. It isolates a mechanism that is hard to identify cleanly in released language-model checkpoints: a feature is present in the input but unused by the base task, the model consolidates, the unused direction becomes less decodable, and a later injected rule that requires that direction becomes harder to acquire or retain.

## Data design

Each sequence contains two independent features:

- `T0/T1`: the **type** feature.
- `C0/C1`: the **class** feature.

Base task:

```text
<bos> T_t C_c Q_BASE Y_T_t <eos>
```

The base label depends only on `T_t`. The class feature is present but behaviourally unused.

Injection task:

```text
<bos> T_t C_c Q_CLASS Y_C_c <eos>
```

The injected label depends on `C_c`, so injection uptake requires preserving or recovering the class direction.

## Schedules

- `S1`: warmup + cosine decay.
- `S2`: warmup + constant learning rate.

The intended claim is read at matched base competence, not merely matched step count.

## Measured quantities

Base trajectory:

- base loss and base accuracy;
- class-rule pre-score;
- stable rank / effective rank summaries;
- linear-probe class decodability from hidden states after the class token.

Injection cells:

- class-rule pre-score;
- post-injection uptake;
- post-washout retention;
- base-task preservation;
- update norm split between embedding/head and deeper transformer weights.

## Expected pattern

Under S1, consolidation should reduce class decodability and reduce class-rule uptake/retention after the consolidation point. Under S2, consolidation should be delayed or weaker, class decodability should remain higher, and injection should stay easier at matched base competence.

## Commands

Smoke test:

```bash
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_smoke.yaml
```

Full first pilot:

```bash
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_s1.yaml
PYTHONPATH=src python scripts/run_toy_mechanism.py --config configs/toy_mechanism_s2.yaml
```

Zip outputs without checkpoints:

```bash
zip -r toy_mechanism_handoff.zip results/toy_mechanism \
  -x "*/checkpoints/*.pt"
```

Keep checkpoint files locally unless we need to debug loading.
