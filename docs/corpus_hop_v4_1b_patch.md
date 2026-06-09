# Corpus HOP v4.1b patch

This patch responds to the first real BabyLM corpus build failure.

Observed failure pattern:
- Probe split was all `S` markers, making the value-from-position baseline nearly perfect.
- Dominant transcript/metadata carriers such as `CHI:`, `MOT:`, `URS:` and CHILDES file identifiers were accepted by the heuristic extractor.
- The selected held-out frame leaked back into training when the previous sampler fell back to the full candidate pool.
- Verb/marker positions were concentrated in early sentence positions, making position-only placement baselines too strong.

Changes:
- Reject transcript/speaker/file-id artifacts before extraction.
- Use a stricter single-verb heuristic extractor.
- Reject lowercase action fragments and colon/speaker-like sentences.
- Enforce exact S/P balance in train and probe.
- Choose a held-out frame only if it leaves enough balanced non-heldout training material.
- Never fall back to held-out-frame examples for training.
- Add explicit marker-balance validation and opposite-attractor-size reporting.
- Add candidate/train/probe marker counts to the report.

If the real BabyLM corpus still fails after this patch, increase `corpus.max_sentences`
or reduce `wordhop.n_train` / `wordhop.n_probe` for the first corpus validation pass.
