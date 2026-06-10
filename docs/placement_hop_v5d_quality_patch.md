# Placement-HOP v5d quality and sampler patch

This patch addresses the v5c strict-clean failure where the automated shortcut gate failed only because the selected probe still concentrated too much mass in a few marker positions:

- WORDHOP mode-slot baseline was slightly above threshold.
- WORDHOP frame+length baseline was slightly above threshold.

It also tightens the scientific-probe extraction after manual audit found noun-like false positives such as plural common nouns tagged as verbs.

## Changes

1. **Stricter verb extraction**
   - Requires a direct dependency subject by default.
   - Disallows inherited subjects for coordinated verbs unless explicitly enabled.
   - Restricts accepted verb dependency labels in the strict config.
   - Rejects trailing abbreviation fragments such as sentences ending in `Mr.`.

2. **Stronger split sampler**
   - Increases random split trials.
   - Adds stronger penalties near the hard placement thresholds.
   - Penalises marker-slot concentration in the probe.
   - Penalises frame/length bucket concentration in the probe.

3. **Strict-clean config**
   - Adds `configs/placement_hop_v5d_babylm_strict_clean.yaml`.
   - Uses the clean written BabyLM sources.
   - Defaults to `n_train: 800`, `n_probe: 200`, because stricter filtering intentionally prioritises quality over size.

The hard validation thresholds are unchanged.
