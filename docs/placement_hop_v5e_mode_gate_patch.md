# Placement-HOP v5e mode-slot patch

This patch targets the remaining v5d failure where the WORDHOP mode-slot baseline was 0.160, just above the 0.15 hard gate. The data were otherwise clean: length-anchored, frame+length, adversarial position-only, naturalness, and carrier-overlap gates passed.

Changes:

- Increase split-selection trials while reusing the cached spaCy candidate pool.
- Penalize probe marker-slot concentration more strongly.
- Add a greedy post-selection repair step that swaps overrepresented marker slots out of the probe when it improves the full placement-gate score.
- Keep the same hard thresholds; no validation threshold is relaxed.

Run:

```bash
PYTHONPATH=src python scripts/build_placement_hop_v5.py \
  --config configs/placement_hop_v5e_babylm_strict_clean.yaml
```
