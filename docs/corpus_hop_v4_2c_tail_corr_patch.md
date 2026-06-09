# Corpus-HOP v4.2c tail/verb-correlation patch

The v4.2b BabyLM spaCy validation run had one remaining automated failure:

- WORDHOP C4 tail/verb correlation was above the 0.20 gate.

This patch keeps all v4.2 gates unchanged and changes only split selection. It strengthens the tail/verb-correlation penalty, adds verb-index buckets to the diverse sampler, and adds a conservative local same-value swap refinement for the probe split. The refinement preserves:

- probe size;
- exact S/P balance;
- held-out-frame count;
- source disjointness;
- labels and transforms.

The goal is to choose a less shortcut-prone probe from the same parser-valid BabyLM candidate pool, not to relax the acceptance gate.
