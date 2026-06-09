# Corpus-HOP v4.1c patch

This patch targets the two remaining BabyLM corpus-HOP failures observed after v4.1b:

1. WORDHOP mode-slot baseline slightly above threshold.
2. WORDHOP marker-index / sentence-length correlation above threshold.

Changes:

- The corpus-derived probe split is now selected before the training split, because the v4.1 acceptance gate is defined on the held-out probe distribution. Training is then drawn from remaining non-heldout sources.
- Probe selection minimises the same shortcut signals used by the gate: marker-slot concentration, marker-length correlation, value-position correlations, and tail/verb correlation.
- Sampling buckets now include the actual WORDHOP marker slot and tail-length bucket, not only verb position and source length.
- Report metadata now includes pre-transform train/probe position metrics for easier debugging.

This patch does not loosen any threshold. It only changes which valid corpus examples are selected.
