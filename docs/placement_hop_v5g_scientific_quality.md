# Placement-HOP v5g: scientific-quality SimpleWiki probe

v5f SimpleWiki passed the automated placement gate, but the audit still contained noisy carriers and ambiguous verb targets. v5g keeps the neutral-marker placement-only design and tightens the extraction quality for the final LM-tier probe.

Main changes:

- uses SimpleWiki only;
- keeps only clear `VBZ` present-tense lexical verbs;
- requires the verb lemma to differ from the surface form;
- requires a direct subject;
- rejects parenthetical/bracketed sentences, semicolons, apostrophes/contractions, dialogue quotes, and known noisy SimpleWiki fragments;
- lowers maximum sentence length to 42 tokens;
- keeps the same placement shortcut gates and naturalness gate.

This deliberately narrows the carrier distribution. That is acceptable for the first Pythia/BabyLM placement probe because the scientific readout is counted marker placement, and clean verb identity is more important than broad coverage.

Run:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5g_babylm_simplewiki_scientific.yaml
```

If there are too few candidates:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5g_babylm_simplewiki_scientific_small.yaml
```
