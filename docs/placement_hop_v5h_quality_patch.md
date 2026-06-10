# Placement-HOP v5h quality patch

v5g passed the automated placement gates but the audit still exposed a few corpus-quality issues, especially title/list-like SimpleWiki lines and colon/list fragments. v5h keeps the neutral-marker placement-only design and the same placement gates, but adds stricter sentence-level quality filters:

- reject colon/list sentences;
- reject comma-without-space fragments;
- reject Roman-numeral heading fragments such as `Act I ...`;
- require at least one common function/stop word to remove title/list rows;
- exclude the known noun-like false positive `population`.

The goal is not to change the scientific target, but to make the manual verb/lemma audit cleaner before freezing the placement probe for Pythia/BabyLM-tier use.
