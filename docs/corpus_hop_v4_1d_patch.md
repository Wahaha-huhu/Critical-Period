# Corpus HOP v4.1d patch

This patch targets the final residual BabyLM corpus validation failure seen after v4.1c: WORDHOP C4 tail/verb correlation slightly above threshold.

Changes:

- Strengthens the probe sampler objective for `tail_verb_corr`.
- Adds an explicit excess penalty once tail/verb correlation exceeds 0.19.
- Increases selection trials for probe/train sampling.
- Does not relax any validation threshold.

If this still fails on the local BabyLM corpus, reduce `n_probe` to 200 or switch to a parser-backed extractor to enlarge the usable candidate pool.
