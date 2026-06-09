# Injection generator v4

This version strengthens the controlled injection datasets before any BabyLM-scale compute is used.

## Structural changes

- WORDHOP source sentences now have variable verb positions rather than a small set of fixed positions.
- Post-verb tail length is sampled independently and widely, so sentence length does not reveal the marker slot.
- Punctuation is inserted inside the four-word counting window on a controlled fraction of items, making WORDHOP and TOKENHOP diverge.
- Probe items include seen-frame and held-out-frame examples.
- Attractor count varies across zero, one, and two. Training includes same-number and opposite-number attractors.
- Metadata records verb index, marker index, head noun, attractors, frame shape, punctuation condition, tail length, and split type.

## Factual changes

- Each injected fact unit has two statements: mineral → discoverer and discoverer → birth town.
- The birth town differs from the identification town.
- No single training sentence contains both the mineral and the compositional target.
- Fictional names no longer share visible numeric suffixes.
- Each probe depth has at least three prompt templates, with held-out template annotations for semantic and compositional probes.

## Strengthened validation gate

The report now includes position-only placement baselines for WORDHOP:

- global mode slot
- length-anchored slot
- frame and length bucket slot
- verb-relative oracle

The first three must fail and the verb-relative oracle must succeed. The report also includes value-shortcut baselines on opposite-attractor probes, TOKENHOP/WORDHOP divergence, held-out frame integrity, and factual two-hop non-leak checks.
