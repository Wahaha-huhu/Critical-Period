# LM-tier injection protocol for the frozen v5h placement probe

## Scope

This protocol freezes the language-model-tier structural probe after the v5h SimpleWiki validation pass. The structural readout is neutral-marker placement, not marker value. The goal is to test whether a model can learn to locate a lexical present-tense verb and place a neutral marker at a counted offset.

## Frozen structural arms

- `WORDHOP`: lemmatise the target verb and insert `<HOP>` four words after the verb, skipping punctuation.
- `NOHOP`: lemmatise the same target verb and insert `<HOP>` immediately after the verb.
- WORDHOP and NOHOP must share identical source carriers in train and probe.
- Train and probe carriers must be disjoint.

## Interpretive floor

The structural dataset is accepted only if position-only predictors cannot place the marker. The v5h audit/comment establishes an exact-slot position-only floor near 0.15. Therefore, model placement accuracy meaningfully above 0.15 on the held-out probe should not be attributed to a simple absolute-position shortcut.

Report, at minimum:

- mode-slot exact accuracy;
- length-anchored exact accuracy;
- frame/length-bucket exact accuracy;
- adversarial position-only exact accuracy;
- verb-relative oracle exact accuracy;
- marker-length correlation as a diagnostic, not a hard gate.

## Pythia `<HOP>` token handling

`<HOP>` is not guaranteed to be a single token in pretrained tokenizers. For Pythia, use the following frozen handling:

1. Add `<HOP>` as an additional special token.
2. Resize the input embedding and language-model head.
3. Initialise the new row with the mean of existing token embeddings unless a run manifest explicitly states another pre-declared initialisation.
4. Pre-score immediately after resizing, before any injection update.
5. During injection, log:
   - `<HOP>` embedding update norm;
   - `<HOP>` embedding gradient norm;
   - transformer-block gradient norm;
   - embedding/deep-gradient ratio.
6. Run one embedding-only ablation on one checkpoint if the main uptake is strong, to check whether placement learning is deeper than learning a new marker row.

## Injection cell

For each checkpoint and each arm:

1. Load checkpoint.
2. Add `<HOP>` token and resize embeddings.
3. Pre-score on the held-out probe.
4. Inject with a fresh optimiser at a fixed learning rate and fixed dose.
5. Score uptake immediately after injection.
6. If uptake is meaningful, run retention by continuing on the frozen natural washout split with no injected examples.
7. Score retention on the same held-out probe.

All cells must use the same injection dose, injection rate, batch size, washout text, washout length, and washout rate within a comparison grid.

## Placement metrics

Primary:

- exact placement accuracy: whether `<HOP>` has highest probability at the true slot among candidate slots;
- placement margin: `log p(<HOP> at true slot) - max log p(<HOP> at incorrect candidate slots)`.

Secondary:

- NOHOP placement uptake;
- WORDHOP minus NOHOP specificity gap;
- punctuation-window subset performance;
- retention fraction after washout.

## Factual control

Use a lightweight memorisation-only fictional fact control. Its job is not to test compositional reasoning. It only checks whether a factual signal injected at the same rate and dose is less stage-sensitive than WORDHOP placement.

Required files:

- `facts_train.jsonl`: injected factual statements;
- `facts_probe.jsonl`: memorisation probes only.

## Washout split

Retention uses a frozen natural SimpleWiki washout split disjoint from all WORDHOP/NOHOP train and probe carriers. The washout split is unmarked natural text only.

## Manual audit

Before injection, complete at least 30 rows in the verb/lemma audit file. Record:

- verb is a genuine lexical verb;
- lemma is correct;
- carrier sentence is acceptable natural text;
- WORDHOP slot is correct;
- fatal error flag and notes.

Keep the completed audit with thesis artifacts.
