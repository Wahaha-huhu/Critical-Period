from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable, Literal

PUNCT = {",", ".", ";", ":", "!", "?"}
HopKind = Literal["NOHOP", "TOKENHOP", "WORDHOP"]


def is_word_token(tok: str) -> bool:
    return tok not in PUNCT


def detokenize(tokens: list[str]) -> str:
    text = ""
    for tok in tokens:
        if not text:
            text = tok
        elif tok in PUNCT:
            text += tok
        else:
            text += " " + tok
    return text


def tokenize_simple(sentence: str) -> list[str]:
    # Split punctuation into separate tokens. This is deliberately simple and
    # deterministic because the generated sentences use a restricted style.
    return re.findall(r"[A-Za-z0-9_<>]+|[,.;:!?]", sentence)


@dataclass(frozen=True)
class SourceSentence:
    source_id: str
    tokens: list[str]
    verb_index: int
    verb_inflected: str
    verb_lemma: str
    marker: str
    subject_number: str
    split: str
    template: str
    has_attractor: bool
    attractor_number: str | None = None

    @property
    def source_text(self) -> str:
        return detokenize(self.tokens)


def _marker_insert_index(tokens_after_lemma: list[str], verb_index: int, hop: HopKind, hop_distance: int) -> int:
    if hop == "NOHOP":
        return verb_index + 1
    if hop == "TOKENHOP":
        return min(len(tokens_after_lemma), verb_index + 1 + hop_distance)
    if hop == "WORDHOP":
        count = 0
        for idx in range(verb_index + 1, len(tokens_after_lemma)):
            if is_word_token(tokens_after_lemma[idx]):
                count += 1
                if count == hop_distance:
                    return idx + 1
        raise ValueError(
            f"Sentence has fewer than {hop_distance} word tokens after verb: {detokenize(tokens_after_lemma)}"
        )
    raise ValueError(f"Unknown hop: {hop}")


def transform_sentence(src: SourceSentence, hop: HopKind, hop_distance: int = 4) -> dict:
    """Lemmatise the verb and insert the S/P marker according to the hop rule."""
    tokens = list(src.tokens)
    if tokens[src.verb_index] != src.verb_inflected:
        raise ValueError(f"Verb mismatch in {src.source_id}: expected {src.verb_inflected} at index {src.verb_index}")
    tokens[src.verb_index] = src.verb_lemma
    insert_index = _marker_insert_index(tokens, src.verb_index, hop, hop_distance)
    tokens_with_marker = tokens[:insert_index] + [src.marker] + tokens[insert_index:]
    wrong_marker = "P" if src.marker == "S" else "S"
    wrong_tokens = tokens[:insert_index] + [wrong_marker] + tokens[insert_index:]
    return {
        "id": f"{src.source_id}__{hop.lower()}",
        "source_id": src.source_id,
        "split": src.split,
        "arm": hop,
        "hop_distance": hop_distance if hop != "NOHOP" else 0,
        "source_text": src.source_text,
        "text": detokenize(tokens_with_marker),
        "tokens": tokens_with_marker,
        "correct_text": detokenize(tokens_with_marker),
        "incorrect_text": detokenize(wrong_tokens),
        "verb_index_original": src.verb_index,
        "verb_index_transformed": src.verb_index,
        "marker_index": insert_index,
        "marker": src.marker,
        "wrong_marker": wrong_marker,
        "subject_number": src.subject_number,
        "verb_lemma": src.verb_lemma,
        "verb_inflected": src.verb_inflected,
        "template": src.template,
        "has_attractor": src.has_attractor,
        "attractor_number": src.attractor_number,
        "metric_targets": {
            "marker_value_margin": {"correct": src.marker, "incorrect": wrong_marker, "position": insert_index},
            "placement": {"correct_position": insert_index, "marker": src.marker},
        },
    }


SINGULAR_SUBJECTS = [
    "author", "engineer", "artist", "teacher", "doctor", "pilot", "farmer", "judge",
    "writer", "river", "painting", "cabinet", "machine", "student", "singer", "chef",
]
PLURAL_SUBJECTS = [
    "authors", "engineers", "artists", "teachers", "doctors", "pilots", "farmers", "judges",
    "writers", "rivers", "paintings", "cabinets", "machines", "students", "singers", "chefs",
]
SINGULAR_ATTRACTORS = ["battle", "desk", "village", "garden", "engine", "journal", "library", "window"]
PLURAL_ATTRACTORS = ["battles", "desks", "villages", "gardens", "engines", "journals", "libraries", "windows"]
VERBS = [
    ("write", "writes"), ("repair", "repairs"), ("paint", "paints"), ("read", "reads"),
    ("hold", "holds"), ("hang", "hangs"), ("flow", "flows"), ("carry", "carries"),
    ("watch", "watches"), ("visit", "visits"), ("study", "studies"), ("clean", "cleans"),
]
AFTER_PHRASES = [
    ["detailed", "notes", "in", "her", "journal"],
    ["the", "broken", "engine", "before", "departure"],
    ["past", "the", "quiet", "village", "daily"],
    ["the", "assigned", "chapters", "every", "week"],
    ["above", "the", "stone", "fireplace", "inside"],
    ["the", "old", "paper", "files", "carefully"],
    [",", "slowly", "and", "carefully", ",", "every", "morning"],
    ["near", "the", "small", "wooden", "bridge"],
]
PREPS = ["of", "beside", "near", "behind", "around", "with"]


def _make_source(
    idx: int,
    rng: random.Random,
    split: str,
    with_attractor: bool,
    opposite_attractor: bool,
) -> SourceSentence:
    subject_number = "singular" if idx % 2 == 0 else "plural"
    if subject_number == "singular":
        subject = rng.choice(SINGULAR_SUBJECTS)
        marker = "S"
        verb_lemma, verb_s = rng.choice(VERBS)
        verb = verb_s
        attr_same_pool, attr_opp_pool = SINGULAR_ATTRACTORS, PLURAL_ATTRACTORS
    else:
        subject = rng.choice(PLURAL_SUBJECTS)
        marker = "P"
        verb_lemma, verb_s = rng.choice(VERBS)
        verb = verb_lemma
        attr_same_pool, attr_opp_pool = PLURAL_ATTRACTORS, SINGULAR_ATTRACTORS

    det = "The"
    tokens = [det, subject]
    template = "plain"
    has_attractor = False
    attr_num = None
    if with_attractor:
        has_attractor = True
        prep = rng.choice(PREPS)
        if opposite_attractor:
            attractor = rng.choice(attr_opp_pool)
            attr_num = "plural" if subject_number == "singular" else "singular"
            template = "attractor_opposite"
        else:
            attractor = rng.choice(attr_same_pool)
            attr_num = subject_number
            template = "attractor_same"
        tokens += [prep, "the", attractor]
    tokens.append(verb)
    after = list(rng.choice(AFTER_PHRASES))
    # Ensure at least four word tokens after the verb.
    if sum(1 for t in after if is_word_token(t)) < 4:
        after += ["today", "outside", "again", "quietly"]
    tokens += after + ["."]
    verb_index = len(tokens) - len(after) - 1 - 1  # before after phrase and final period
    # Simpler and safer: find the unique just-appended verb before the after phrase.
    verb_index = tokens.index(verb, 2)
    return SourceSentence(
        source_id=f"{split}_{idx:06d}",
        tokens=tokens,
        verb_index=verb_index,
        verb_inflected=verb,
        verb_lemma=verb_lemma,
        marker=marker,
        subject_number=subject_number,
        split=split,
        template=template,
        has_attractor=has_attractor,
        attractor_number=attr_num,
    )


def _source_signature(src: SourceSentence) -> tuple:
    """Signature used to keep train and probe structurally disjoint.

    It intentionally ignores source_id and split and records the actual source
    text plus the target marker. If two source items share this signature, their
    transformed NOHOP/TOKENHOP/WORDHOP examples can overlap exactly.
    """
    return (tuple(src.tokens), src.verb_index, src.verb_lemma, src.marker)


def generate_sources(
    n_train: int,
    n_probe: int,
    seed: int,
    attractor_probe_fraction: float = 0.5,
) -> tuple[list[SourceSentence], list[SourceSentence]]:
    rng = random.Random(seed)
    train: list[SourceSentence] = []
    probe: list[SourceSentence] = []
    train_sigs: set[tuple] = set()
    probe_sigs: set[tuple] = set()

    attempts = 0
    i = 0
    while len(train) < n_train:
        attempts += 1
        if attempts > n_train * 200:
            raise RuntimeError("Could not generate enough unique training sources")
        # Training includes mostly plain sentences plus some same-number attractors.
        with_attr = (i % 5 == 0)
        src = _make_source(len(train), rng, "train", with_attr, opposite_attractor=False)
        sig = _source_signature(src)
        i += 1
        if sig in train_sigs:
            continue
        train_sigs.add(sig)
        train.append(src)

    attempts = 0
    i = 0
    while len(probe) < n_probe:
        attempts += 1
        if attempts > n_probe * 500:
            raise RuntimeError("Could not generate enough unique held-out probe sources")
        with_attr = (len(probe) / max(1, n_probe)) < attractor_probe_fraction
        opposite = with_attr and (len(probe) % 2 == 0)
        src = _make_source(len(probe), rng, "probe", with_attr, opposite_attractor=opposite)
        sig = _source_signature(src)
        i += 1
        if sig in train_sigs or sig in probe_sigs:
            continue
        probe_sigs.add(sig)
        probe.append(src)
    return train, probe


def build_wordhop_dataset(
    n_train: int = 2000,
    n_probe: int = 500,
    seed: int = 0,
    hop_distance: int = 4,
    include_tokenhop: bool = True,
) -> dict[str, list[dict]]:
    train_src, probe_src = generate_sources(n_train=n_train, n_probe=n_probe, seed=seed)
    arms: list[HopKind] = ["NOHOP", "WORDHOP"]
    if include_tokenhop:
        arms.insert(1, "TOKENHOP")
    out: dict[str, list[dict]] = {}
    for arm in arms:
        out[f"{arm.lower()}_train"] = [transform_sentence(s, arm, hop_distance) for s in train_src]
        out[f"{arm.lower()}_probe"] = [transform_sentence(s, arm, hop_distance) for s in probe_src]
    return out
