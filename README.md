# Critical-period LM experiments — v5f placement-HOP quality patch

This repository contains the current experiment code for the critical-period/plasticity project.

The newest path is **Placement-HOP v5f**:

- neutral marker `<HOP>`;
- structural readout is placement only;
- SimpleWiki-only corpus validation config for cleaner natural carriers;
- CPU-friendly spaCy candidate cache and `max_candidates` stopping;
- positional/adversarial placement shortcut gates;
- verb/lemma audit export.

Run the recommended validation:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5f_babylm_simplewiki.yaml
```

Zip the result:

```bash
zip -r placement_hop_v5f_simplewiki_validation.zip \
  results/dataset_validation/placement_hop_v5f_babylm_simplewiki
```

Fallback if there are too few candidates:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5f_babylm_simplewiki_small.yaml
```

See `docs/placement_hop_v5f_simplewiki.md` for details.
