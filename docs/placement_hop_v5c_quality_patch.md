# Placement-HOP v5c quality patch

v5b showed that the automated placement shortcut gate can pass while the selected carrier sentences still contain dialogue, markup headings, multi-sentence fragments, or POS false positives. v5c keeps the same neutral-marker placement-only design, but adds stricter scientific-probe filters for the LM-tier dataset:

- reject wiki heading/markup remnants containing `=`;
- optionally reject double-quoted dialogue sentences;
- optionally reject sentences beginning with quotes;
- optionally reject multi-sentence fragments by terminal punctuation count;
- reduce max commas in the clean config;
- write the same verb/lemma audit CSV as before.

The gates are unchanged: position-only placement baselines and adversarial placement must fail, the verb-relative oracle must pass, naturalness must pass, and train/probe carriers must be disjoint. The stricter filters are intended to improve manual audit quality, not to make the statistical gate easier.
