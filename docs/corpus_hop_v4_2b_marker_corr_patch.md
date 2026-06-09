# Corpus HOP v4.2b marker-length decorrelation patch

The v4.2 spaCy BabyLM build reduced parser noise and passed the placement baselines, value-position gate, naturalness proxy, held-out-frame integrity, and factual checks, but one run still failed C3 because WORDHOP marker index was correlated with source sentence length.

This patch does not relax the gate. It strengthens the split sampler so probe selection gives much higher priority to lowering marker-index/source-length correlation while preserving S/P balance, held-out-frame disjointness, low mode-slot accuracy, and the existing value-position constraints.

Changes:

- lowers the sampler soft threshold for marker-length correlation from 0.28 to 0.24;
- increases the direct weight on marker-length correlation;
- adds a large linear and quadratic excess penalty above the soft threshold;
- increases low-leak split-selection trials for probe and train selection.

Use the same config as v4.2 and rerun `scripts/build_corpus_hop_datasets.py`.
