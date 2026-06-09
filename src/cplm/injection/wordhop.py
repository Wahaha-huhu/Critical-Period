from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Literal

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
    head_noun_index: int | None = None
    head_noun: str | None = None
    attractor_indices: list[int] = field(default_factory=list)
    attractor_numbers: list[str] = field(default_factory=list)
    attractor_count: int = 0
    frame_shape: str = "unknown"
    frame_split_type: str = "seen_frame"
    punctuation_condition: str = "none"
    window_punctuation_count: int = 0
    tail_word_length: int = 0
    source_length: int = 0
    metadata: dict = field(default_factory=dict)

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


def _tail_word_length_after_marker(tokens_with_marker: list[str], marker_index: int) -> int:
    return sum(1 for tok in tokens_with_marker[marker_index + 1 :] if is_word_token(tok))


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
        "frame_split_type": src.frame_split_type,
        "arm": hop,
        "hop_distance": hop_distance if hop != "NOHOP" else 0,
        "source_text": src.source_text,
        "text": detokenize(tokens_with_marker),
        "tokens": tokens_with_marker,
        "source_tokens": src.tokens,
        "correct_text": detokenize(tokens_with_marker),
        "incorrect_text": detokenize(wrong_tokens),
        "verb_index_original": src.verb_index,
        "verb_index_transformed": src.verb_index,
        "marker_index": insert_index,
        "marker": src.marker,
        "wrong_marker": wrong_marker,
        "subject_number": src.subject_number,
        "head_noun_index": src.head_noun_index,
        "head_noun": src.head_noun,
        "verb_lemma": src.verb_lemma,
        "verb_inflected": src.verb_inflected,
        "template": src.template,
        "frame_shape": src.frame_shape,
        "has_attractor": src.has_attractor,
        "attractor_number": src.attractor_number,
        "attractor_indices": src.attractor_indices,
        "attractor_numbers": src.attractor_numbers,
        "attractor_count": src.attractor_count,
        "punctuation_condition": src.punctuation_condition,
        "window_punctuation_count": src.window_punctuation_count,
        "source_length": len(src.tokens),
        "marked_length": len(tokens_with_marker),
        "tail_word_length_after_marker": _tail_word_length_after_marker(tokens_with_marker, insert_index),
        "training_opposite_attractor_policy": "mixed: train includes 0/1/2 attractors with both same-number and opposite-number cases",
        "metric_targets": {
            "marker_value_margin": {"correct": src.marker, "incorrect": wrong_marker, "position": insert_index},
            "placement": {"correct_position": insert_index, "marker": src.marker},
        },
    }


SINGULAR_HEADS = [
    "author", "engineer", "artist", "teacher", "doctor", "pilot", "farmer", "judge", "writer", "river",
    "painting", "cabinet", "machine", "student", "singer", "chef", "scientist", "gardener", "archivist",
    "merchant", "captain", "librarian", "actor", "composer", "visitor", "minister", "carpenter", "traveler",
]
PLURAL_HEADS = [
    "authors", "engineers", "artists", "teachers", "doctors", "pilots", "farmers", "judges", "writers", "rivers",
    "paintings", "cabinets", "machines", "students", "singers", "chefs", "scientists", "gardeners", "archivists",
    "merchants", "captains", "librarians", "actors", "composers", "visitors", "ministers", "carpenters", "travelers",
]
SINGULAR_ATTRACTORS = [
    "battle", "desk", "village", "garden", "engine", "journal", "library", "window", "statue", "museum", "harbor",
    "cottage", "bridge", "forest", "school", "theater", "island", "letter", "lantern", "valley", "camera",
]
PLURAL_ATTRACTORS = [
    "battles", "desks", "villages", "gardens", "engines", "journals", "libraries", "windows", "statues", "museums", "harbors",
    "cottages", "bridges", "forests", "schools", "theaters", "islands", "letters", "lanterns", "valleys", "cameras",
]
ADJECTIVES = [
    "quiet", "careful", "young", "old", "curious", "patient", "bright", "distant", "famous", "modest", "restless",
    "skillful", "local", "ancient", "small", "large", "wooden", "silver", "narrow", "hidden", "formal", "gentle",
]
VERBS = [
    ("write", "writes"), ("repair", "repairs"), ("paint", "paints"), ("read", "reads"), ("hold", "holds"),
    ("hang", "hangs"), ("flow", "flows"), ("carry", "carries"), ("watch", "watches"), ("visit", "visits"),
    ("study", "studies"), ("clean", "cleans"), ("admire", "admires"), ("measure", "measures"),
    ("describe", "describes"), ("follow", "follows"), ("collect", "collects"), ("inspect", "inspects"),
]
PREPS = ["of", "beside", "near", "behind", "around", "with", "inside", "beyond", "under", "over"]
WINDOW_WORDS = [
    "detailed", "notes", "beside", "her", "journal", "broken", "engine", "before", "departure", "quiet", "village",
    "assigned", "chapters", "every", "week", "stone", "fireplace", "paper", "files", "carefully", "wooden", "bridge",
    "market", "during", "autumn", "morning", "across", "narrow", "street", "inside", "private", "office",
]
TAIL_WORDS = [
    "today", "outside", "again", "quietly", "nearby", "afterward", "often", "there", "alone", "indoors", "patiently",
    "before", "sunset", "during", "winter", "without", "delay", "beside", "water", "yesterday", "tomorrow",
]
FRAME_FAMILIES = ["simple", "pp1", "pp2", "rel", "pp1_rel", "pp2_rel"]
TRAIN_FRAME_FAMILIES = ["simple", "pp1", "pp2", "rel", "pp1_rel"]
HELDOUT_FRAME_FAMILIES = ["pp2_rel"]


def _noun_for_number(rng: random.Random, number: str, role: str = "head") -> str:
    if role == "head":
        return rng.choice(SINGULAR_HEADS if number == "singular" else PLURAL_HEADS)
    return rng.choice(SINGULAR_ATTRACTORS if number == "singular" else PLURAL_ATTRACTORS)


def _article_for(noun: str, plural: bool = False) -> str:
    if plural:
        return "the"
    return "an" if noun[0].lower() in "aeiou" else "the"


def _subject_tokens(
    rng: random.Random,
    subject_number: str,
    frame_family: str,
    attractor_count: int,
    opposite_policy: str,
) -> tuple[list[str], int, str, list[int], list[str]]:
    tokens: list[str] = ["The"]
    # Adjective count is deliberately not encoded in frame_shape. This creates
    # broad verb-index spread inside each structural frame.
    for _ in range(rng.randint(0, 3)):
        tokens.append(rng.choice(ADJECTIVES))
    head = _noun_for_number(rng, subject_number, "head")
    head_idx = len(tokens)
    tokens.append(head)
    attractor_indices: list[int] = []
    attractor_numbers: list[str] = []

    max_pp = 2 if frame_family in {"pp2", "pp2_rel"} else 1 if frame_family in {"pp1", "pp1_rel"} else 0
    rel = frame_family in {"rel", "pp1_rel", "pp2_rel"}
    # Match requested attractor count while respecting frame capacity.
    capacity = max_pp + (1 if rel else 0)
    n_attr = min(attractor_count, capacity)
    remaining_opposite = 0
    if opposite_policy == "opposite" and n_attr > 0:
        # All realized attractors are opposite-number in strict opposite probes.
        # This makes nearest-noun and last-attractor shortcuts fail decisively.
        remaining_opposite = n_attr
    elif opposite_policy == "mixed" and n_attr > 0:
        remaining_opposite = rng.randint(0, n_attr)

    def add_attractor_phrase(prefix: list[str]) -> None:
        nonlocal remaining_opposite
        use_opp = remaining_opposite > 0
        if use_opp:
            attr_num = "plural" if subject_number == "singular" else "singular"
            remaining_opposite -= 1
        else:
            attr_num = subject_number
        noun = _noun_for_number(rng, attr_num, "attr")
        phrase = list(prefix)
        phrase.append(_article_for(noun, plural=(attr_num == "plural")))
        for _ in range(rng.randint(0, 2)):
            phrase.append(rng.choice(ADJECTIVES))
        phrase.append(noun)
        tokens.extend(phrase)
        attractor_indices.append(len(tokens) - 1)
        attractor_numbers.append(attr_num)

    for _ in range(max_pp):
        if len(attractor_indices) < n_attr:
            add_attractor_phrase([rng.choice(PREPS)])
        else:
            # Non-attractor PP with same-number or neutral noun still adds length.
            noun_num = subject_number
            noun = _noun_for_number(rng, noun_num, "attr")
            tokens += [rng.choice(PREPS), _article_for(noun, plural=(noun_num == "plural")), noun]
    if rel:
        if len(attractor_indices) < n_attr:
            add_attractor_phrase(["that", "stood", rng.choice(["near", "beside", "behind", "around"])])
        else:
            tokens += ["that", rng.choice(["arrived", "waited", "returned", "rested"])]
    return tokens, head_idx, head, attractor_indices, attractor_numbers


def _after_verb_tokens(rng: random.Random, punctuation_condition: str, tail_len: int) -> tuple[list[str], int]:
    first_four = rng.sample(WINDOW_WORDS, 4)
    tokens: list[str] = []
    punc_count = 0
    for i, word in enumerate(first_four):
        tokens.append(word)
        if punctuation_condition == "window_punct" and i in {0, 1, 2} and rng.random() < 0.55:
            tokens.append(rng.choice([",", ";"]))
            punc_count += 1
    # Guarantee divergence in a controlled majority when requested.
    if punctuation_condition == "window_punct" and punc_count == 0:
        tokens.insert(2, ",")
        punc_count = 1
    tail = rng.sample(TAIL_WORDS, k=min(tail_len, len(TAIL_WORDS)))
    # If k cannot cover requested length, extend deterministically.
    while len(tail) < tail_len:
        tail.append(rng.choice(TAIL_WORDS))
    tokens.extend(tail)
    if rng.random() < 0.25 and tail_len > 1:
        tokens.insert(len(tokens) - rng.randint(1, min(3, len(tokens) - 1)), ",")
    tokens.append(".")
    return tokens, punc_count


def _make_source(
    idx: int,
    rng: random.Random,
    split: str,
    frame_family: str,
    frame_split_type: str,
    subject_number: str,
    attractor_count: int,
    opposite_policy: str,
    punctuation_condition: str,
    tail_len: int,
) -> SourceSentence:
    marker = "S" if subject_number == "singular" else "P"
    verb_lemma, verb_s = rng.choice(VERBS)
    verb = verb_s if subject_number == "singular" else verb_lemma
    subj_tokens, head_idx, head, attr_indices, attr_nums = _subject_tokens(
        rng, subject_number, frame_family, attractor_count, opposite_policy
    )
    tokens = list(subj_tokens)
    verb_index = len(tokens)
    tokens.append(verb)
    after, punc_count = _after_verb_tokens(rng, punctuation_condition, tail_len)
    tokens.extend(after)
    has_opp = any(n != subject_number for n in attr_nums)
    if not attr_nums:
        template = "plain"
    elif has_opp:
        template = "attractor_opposite"
    else:
        template = "attractor_same"
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
        has_attractor=bool(attr_nums),
        attractor_number=attr_nums[-1] if attr_nums else None,
        head_noun_index=head_idx,
        head_noun=head,
        attractor_indices=attr_indices,
        attractor_numbers=attr_nums,
        attractor_count=len(attr_nums),
        frame_shape=frame_family,
        frame_split_type=frame_split_type,
        punctuation_condition=punctuation_condition,
        window_punctuation_count=punc_count,
        tail_word_length=tail_len,
        source_length=len(tokens),
    )


def _source_signature(src: SourceSentence) -> tuple:
    return (tuple(src.tokens), src.verb_index, src.verb_lemma, src.marker)


def _sample_specs(n: int, split: str, rng: random.Random) -> list[dict]:
    specs: list[dict] = []
    for i in range(n):
        subject_number = "singular" if i % 2 == 0 else "plural"
        if split == "train":
            frame_family = rng.choice(TRAIN_FRAME_FAMILIES)
            frame_split_type = "train_frame"
        else:
            # Half the probe uses seen frame families, half uses held-out frames.
            if i % 2 == 0:
                frame_family = rng.choice(TRAIN_FRAME_FAMILIES)
                frame_split_type = "seen_frame"
            else:
                frame_family = rng.choice(HELDOUT_FRAME_FAMILIES)
                frame_split_type = "heldout_frame"
        # Ensure 0/1/2 attractor coverage. Capacity is handled by _subject_tokens.
        attractor_count = i % 3
        if split == "probe" and attractor_count > 0:
            opposite_policy = "opposite" if (i // 3) % 2 == 0 else "same"
        elif split == "train" and attractor_count > 0:
            # Include opposite attractors in training, so the opposite-attractor
            # probe is not merely an out-of-distribution nearest-noun test.
            opposite_policy = "opposite" if (i // 5) % 3 == 0 else "same"
        else:
            opposite_policy = "none"
        punctuation_condition = "window_punct" if rng.random() < 0.65 else "no_window_punct"
        # Wide independent tail variation is crucial: otherwise total sentence
        # length leaks marker position because marker_index is verb-relative.
        tail_len = rng.randint(0, 50)
        specs.append(
            dict(
                split=split,
                frame_family=frame_family,
                frame_split_type=frame_split_type,
                subject_number=subject_number,
                attractor_count=attractor_count,
                opposite_policy=opposite_policy,
                punctuation_condition=punctuation_condition,
                tail_len=tail_len,
            )
        )
    return specs


def generate_sources(n_train: int, n_probe: int, seed: int, attractor_probe_fraction: float = 0.5) -> tuple[list[SourceSentence], list[SourceSentence]]:
    # attractor_probe_fraction is kept for backwards compatibility; v4 uses an
    # explicit 0/1/2 attractor cycle instead.
    rng = random.Random(seed)
    train: list[SourceSentence] = []
    probe: list[SourceSentence] = []
    train_sigs: set[tuple] = set()
    probe_sigs: set[tuple] = set()

    def fill(target: list[SourceSentence], n: int, split: str, seen_sigs: set[tuple], forbidden_sigs: set[tuple]) -> None:
        attempts = 0
        i = 0
        specs = _sample_specs(n * 4, split, rng)
        while len(target) < n:
            if i >= len(specs):
                specs.extend(_sample_specs(n, split, rng))
            spec = specs[i]
            i += 1
            attempts += 1
            if attempts > n * 1000:
                raise RuntimeError(f"Could not generate enough unique {split} sources")
            src = _make_source(len(target), rng, **spec)
            sig = _source_signature(src)
            if sig in seen_sigs or sig in forbidden_sigs:
                continue
            seen_sigs.add(sig)
            target.append(src)

    fill(train, n_train, "train", train_sigs, set())
    fill(probe, n_probe, "probe", probe_sigs, train_sigs)
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
