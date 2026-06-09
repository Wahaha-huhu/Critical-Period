# Corpus-HOP v4.2 parser-backed extraction

v4.1d passed the statistical shortcut gates on BabyLM with a reduced probe size, but the manual audit still showed heuristic parse errors such as adverbs/adjectives being treated as subjects or nouns being treated as verbs. v4.2 adds a parser-backed extraction path for BabyLM/Pythia-tier structural data.

## Main rule

For language-model-tier experiments, use:

```yaml
parser:
  backend: spacy
  spacy_model: en_core_web_sm
  require_parser: true
```

The old heuristic extractor remains available only for demo fallback, smoke tests, and debugging. It should not be used for BabyLM/Pythia scientific claims.

## Accepted examples

The spaCy extractor keeps only sentences with:

- exactly one lexical present-tense verb (`VBZ` or `VBP`), excluding auxiliaries;
- exactly one clear `nsubj`/`nsubjpass` dependency before the verb;
- subject number inferred from parser morphology/tag;
- verb form compatible with subject number;
- at least four following words for WORDHOP;
- no transcript labels, file IDs, quote markup, or lower-case fragments.

The structural transform is unchanged: NOHOP inserts the marker after the lemmatised verb; WORDHOP inserts it four words after the verb, skipping punctuation.

## Manual audit remains required

A passing automated report is not sufficient by itself. Before BabyLM injection, inspect `corpus_audit_sheet.md`, especially random WORDHOP probe examples and opposite-attractor examples. The expected audit pattern is:

- the subject is a real syntactic subject, not a leading adverb/adjective;
- the verb is a real present-tense lexical verb, not a noun;
- proper names such as `James` are singular;
- attractors are real intervening nouns where present.

## Install

```bash
pip install -U spacy
python -m spacy download en_core_web_sm
```

Then run:

```bash
PYTHONPATH=src python scripts/build_corpus_hop_datasets.py \
  --config configs/corpus_hop_babylm_v42_spacy.yaml
```
