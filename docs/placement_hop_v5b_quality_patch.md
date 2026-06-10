# Placement-HOP v5b quality patch

The first v5 BabyLM build passed the formal placement shortcut gate, but the manual examples showed too many noisy verb extractions from spoken/transcript-heavy BabyLM files. Examples included nouns or fillers treated as verbs and transcript-like carrier sentences. This is not a placement-baseline problem; it is a corpus-probe quality problem.

v5b keeps the neutral-marker placement-only design, but strengthens extraction quality:

- adds an optional `quality.require_subject` filter so the selected verb must have a visible dependency subject, or inherit one as a coordinated verb;
- excludes common spoken fillers and non-standard pseudo-verbs such as `erm`, `wanna`, `gonna`, and `gotta`;
- rejects sentences with multiple spaced acronym/transcript fragments;
- allows a maximum comma count to avoid very long spoken fragments;
- adds a clean BabyLM config using `gutenberg.train.txt` and `simple_wiki.train.txt` first;
- keeps an all-source strict config as a fallback if the clean-source candidate pool is too small.

The intended run is:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5b_babylm_clean.yaml
```

If the clean config produces too few candidates, use:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5b_babylm_all_sources_strict.yaml
```

Acceptance now requires both the automated placement gates and a manual verb/lemma audit. The automated gate alone is not sufficient.
