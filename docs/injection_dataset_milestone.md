# Injection dataset milestone

This milestone implements Experiment 1 from the revised roadmap only. It does not train BabyLM, run checkpoint injections, or touch Pythia.

## Purpose

The goal is to freeze the probe package before spending training compute. The package contains one structural/routing rule, one within-rule control, and one factual control.

## Structural arm

`WORDHOP` lemmatises a third-person present verb and places the number marker `S` or `P` four words after the verb, skipping punctuation. This is the non-native structural/routing rule.

## Within-rule control

`NOHOP` uses the same marker and same transformed sentence family, but places the marker immediately after the lemmatised verb. It is the near-native reuse baseline.

## Pilot difficulty step

`TOKENHOP` places the marker four tokens after the verb. It is generated for pilot difficulty checks but is not the default final arm.

## Factual arm

The factual arm generates fictional mineral-discovery facts and probes memorisation, semantic rephrasing, and a simple compositional relation.

## Validation gate

The generated package should pass these checks before any BabyLM training begins:

- marker position is correct for NOHOP, TOKENHOP, and WORDHOP;
- WORDHOP counts words rather than punctuation;
- singular and plural markers are balanced;
- attractor probes include same-number and opposite-number attractors;
- factual prompts do not leak the target span;
- scoring sanity fixtures have the expected margin sign.
