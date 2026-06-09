from __future__ import annotations

import glob
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Any

from cplm.injection.wordhop import SourceSentence, transform_sentence, tokenize_simple, detokenize, is_word_token
from cplm.injection.wordhop import SINGULAR_HEADS, PLURAL_HEADS, SINGULAR_ATTRACTORS, PLURAL_ATTRACTORS, ADJECTIVES, VERBS

PUNCT = {",", ".", ";", ":", "!", "?"}
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "each", "every", "many", "several", "some"}
PREPOSITIONS = {"of", "near", "beside", "behind", "around", "with", "inside", "beyond", "under", "over", "in", "on", "by", "for", "from", "through", "between", "among", "along", "across", "before", "after"}
REL_WORDS = {"who", "that", "which"}
FUNCTION = DETERMINERS | PREPOSITIONS | REL_WORDS | {"and", "or", "but", "not", "very", "quite", "rather"}

PRONOUN_NUMBERS = {
    "he": "singular", "she": "singular", "it": "singular",
    "they": "plural",
}
AMBIGUOUS_OR_NONTHIRD_PRONOUNS = {"i", "you", "we", "me", "us", "him", "her", "them"}
BAD_TRANSCRIPT_TOKENS = {
    "chi", "mot", "fat", "bro", "sis", "exp", "inv", "par", "pau", "urs", "adu",
    "childes", "subtlex", "speaker", "xxx", "www", "hv", "lw", "et", "na"
}


EXTRA_VERBS = [
    ("make", "makes"), ("take", "takes"), ("keep", "keeps"), ("give", "gives"), ("find", "finds"),
    ("bring", "brings"), ("show", "shows"), ("use", "uses"), ("need", "needs"), ("open", "opens"),
    ("close", "closes"), ("move", "moves"), ("turn", "turns"), ("serve", "serves"), ("reach", "reaches"),
]
VERB_PAIRS = list(dict((lemma, infl) for lemma, infl in (VERBS + EXTRA_VERBS)).items())
LEMMA_TO_INFL = {lemma: infl for lemma, infl in VERB_PAIRS}
INFL_TO_LEMMA = {infl: lemma for lemma, infl in VERB_PAIRS}
LEMMA_SET = set(LEMMA_TO_INFL)
INFL_SET = set(INFL_TO_LEMMA)

SINGULAR_NOUNS = set(SINGULAR_HEADS + SINGULAR_ATTRACTORS + ["book", "letter", "room", "story", "child", "bird", "flower", "map", "house", "teacher", "worker", "reader", "visitor"])
PLURAL_NOUNS = set(PLURAL_HEADS + PLURAL_ATTRACTORS + ["books", "letters", "rooms", "stories", "children", "birds", "flowers", "maps", "houses", "teachers", "workers", "readers", "visitors"])

TAIL_PHRASES = [
    ["detailed", "notes", "beside", "the", "window", "during", "the", "morning"],
    ["the", "small", "box", "near", "the", "wooden", "table", "today"],
    ["careful", "records", "for", "the", "local", "library", "each", "week"],
    ["quiet", "messages", "inside", "the", "old", "office", "after", "lunch"],
    ["the", "green", "folder", "across", "the", "narrow", "hall", "again"],
    ["fresh", "samples", "under", "the", "silver", "lamp", "before", "sunset"],
]

@dataclass(frozen=True)
class Rejection:
    sentence: str
    reason: str


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if len(p.strip()) >= 10]


def load_corpus_sentences(globs: list[str], limit: int | None = None) -> list[str]:
    paths: list[str] = []
    for pat in globs:
        paths.extend(glob.glob(pat, recursive=True))
    paths = sorted(set(paths))
    out: list[str] = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        out.extend(split_sentences(text))
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def _is_noisy_corpus_sentence(sentence: str) -> str | None:
    """Reject transcript/provenance artifacts before heuristic parsing.

    The corpus-derived HOP path is for LM-tier natural carriers. Child-language
    corpora and subtitles contain speaker tags, file IDs, coding tokens, and
    malformed fragments; a conservative rejection is better than extracting
    fake subjects such as CHI/MOT/URS or file identifiers.
    """
    low = sentence.lower()
    if any(x in low for x in ["xxx", "childes_", ".cha", "speaker:", "mot:", "chi:", "fat:", "bro:", "urs:", "pau:"]):
        return "transcript_marker"
    if re.search(r"\b[A-Z]{2,}\s*:", sentence):
        return "speaker_label"
    if ":" in sentence:
        return "colon_or_speaker_like"
    if re.search(r"\b[a-z]{2,}[A-Z_]+\b", sentence):
        return "file_or_markup_token"
    # Too many uppercase labels or IDs is a strong sign of transcript metadata.
    caps = re.findall(r"\b[A-Z]{2,}\b", sentence)
    if len(caps) >= 2:
        return "many_allcaps_tokens"
    if re.search(r"\b\w*\d\w*\b", sentence):
        return "digit_or_file_id"
    return None


def _number_of_noun(tok: str) -> str | None:
    t = tok.lower().strip()
    if t in PRONOUN_NUMBERS:
        return PRONOUN_NUMBERS[t]
    if t in AMBIGUOUS_OR_NONTHIRD_PRONOUNS:
        return None
    if t in BAD_TRANSCRIPT_TOKENS:
        return None
    if t in SINGULAR_NOUNS:
        return "singular"
    if t in PLURAL_NOUNS:
        return "plural"
    if t in {"children", "people", "men", "women"}:
        return "plural"
    if len(t) <= 2 or t in FUNCTION or t in LEMMA_SET or t in INFL_SET:
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", tok):
        return None
    # Conservative morphology fallback. Avoid many false plural endings.
    if t.endswith("s") and not t.endswith(("ss", "us", "is", "'s")):
        return "plural"
    # Proper names and ordinary alphabetic nouns default singular.
    if re.fullmatch(r"[a-z]+", t) or (tok[:1].isupper() and tok[1:].islower()):
        return "singular"
    return None


def _subject_segment(tokens: list[str], verb_index: int) -> tuple[int, list[str]]:
    """Return tokens between last strong boundary and the verb."""
    start = 0
    for j in range(verb_index - 1, -1, -1):
        if tokens[j] in {".", "!", "?", ";", ":"}:
            start = j + 1
            break
    return start, tokens[start:verb_index]


def _head_noun_before_verb(tokens: list[str], verb_index: int) -> tuple[int, str, str] | None:
    # Choose the first plausible head in the local pre-verbal subject segment.
    # This keeps PP nouns as attractors rather than switching to nearest noun.
    start, seg = _subject_segment(tokens, verb_index)
    if not seg or len(seg) > 18:
        return None
    lows = [t.lower() for t in seg]
    if any(t in BAD_TRANSCRIPT_TOKENS or t in AMBIGUOUS_OR_NONTHIRD_PRONOUNS for t in lows):
        return None
    if any(re.fullmatch(r"[A-Z]{2,}", t) for t in seg):
        return None
    # Reject fragments dominated by function words before the verb.
    content_like = [t for t in seg if _number_of_noun(t) and t.lower() not in FUNCTION]
    if not content_like:
        return None
    for off, tok in enumerate(seg):
        low = tok.lower()
        if low in PUNCT or low in FUNCTION or low in ADJECTIVES:
            continue
        num = _number_of_noun(tok)
        if num:
            return start + off, tok, num
    return None


def _attractors_between(tokens: list[str], head_idx: int, verb_index: int, subject_number: str) -> tuple[list[int], list[str]]:
    idxs: list[int] = []
    nums: list[str] = []
    for i in range(head_idx + 1, verb_index):
        low = tokens[i].lower()
        if low in PUNCT or low in FUNCTION or low in ADJECTIVES:
            continue
        num = _number_of_noun(tokens[i])
        if num:
            idxs.append(i)
            nums.append(num)
    return idxs, nums


def _frame_shape(tokens: list[str], head_idx: int, verb_index: int) -> str:
    pre = [t.lower() for t in tokens[head_idx + 1 : verb_index]]
    pp_count = sum(1 for t in pre if t in PREPOSITIONS)
    has_rel = any(t in REL_WORDS for t in pre)
    comma = any(t in {",", ";", ":"} for t in tokens)
    return f"pp{min(pp_count,2)}_rel{int(has_rel)}_comma{int(comma)}"


def _punctuation_in_window(tokens: list[str], verb_index: int, hop_distance: int = 4) -> tuple[str, int]:
    count_words = 0
    punct = 0
    for idx in range(verb_index + 1, len(tokens)):
        if tokens[idx] in PUNCT:
            punct += 1
        elif is_word_token(tokens[idx]):
            count_words += 1
            if count_words >= hop_distance:
                break
    return ("window_punct" if punct else "none", punct)




def _spacy_subject_number(tok: Any) -> str | None:
    """Return singular/plural for a spaCy subject head, conservatively.

    This is intentionally stricter than morphology-only extraction because bad
    labels are worse than low yield for the BabyLM/Pythia injection tier.
    """
    low = tok.text.lower().strip()
    if low in AMBIGUOUS_OR_NONTHIRD_PRONOUNS or low in BAD_TRANSCRIPT_TOKENS:
        return None
    if low in PRONOUN_NUMBERS:
        return PRONOUN_NUMBERS[low]
    if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", tok.text):
        return None
    tag = getattr(tok, "tag_", "")
    pos = getattr(tok, "pos_", "")
    morph_num = set(tok.morph.get("Number")) if hasattr(tok, "morph") else set()
    if "Plur" in morph_num or tag in {"NNS", "NNPS"}:
        return "plural"
    if "Sing" in morph_num or tag in {"NN", "NNP"}:
        return "singular"
    # Proper names such as James are usually NNP/Sing in spaCy; keep this as a
    # final guarded fallback to avoid the old heuristic bug where names ending
    # in s became plural.
    if pos == "PROPN" and tok.text[:1].isupper():
        return "singular"
    if pos in {"NOUN", "PRON"}:
        # Conservative morphology fallback only for common nouns, not proper names.
        if low.endswith("s") and not low.endswith(("ss", "us", "is", "'s")):
            return "plural"
        return "singular"
    return None


def _spacy_clean_tokens(doc: Any) -> list[str]:
    return [t.text for t in doc if not getattr(t, "is_space", False)]


def _spacy_candidate_verb_tokens(doc: Any) -> list[tuple[Any, Any, str, str]]:
    """Return parser-confirmed (verb, subject, lemma, verb_number) candidates."""
    cands = []
    for tok in doc:
        if getattr(tok, "is_space", False):
            continue
        # Use lexical present-tense verbs first. Exclude auxiliaries to avoid
        # existential/copular cases like "there is" and "it is" in the first pass.
        if tok.pos_ != "VERB":
            continue
        if tok.tag_ not in {"VBZ", "VBP"}:
            continue
        if tok.lemma_.lower() in {"be", "have", "do"}:
            continue
        subjects = [c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}]
        if len(subjects) != 1:
            continue
        subj = subjects[0]
        if subj.i >= tok.i:
            continue
        if subj.dep_ == "expl" or subj.text.lower() == "there":
            continue
        subj_num = _spacy_subject_number(subj)
        if subj_num is None:
            continue
        # Third-person present agreement compatibility.
        if tok.tag_ == "VBZ" and subj_num != "singular":
            continue
        if tok.tag_ == "VBP" and subj_num != "plural":
            continue
        # Exclude very long dependencies; they are often parse mistakes or too
        # complex for the first single-site BabyLM probe.
        if tok.i - subj.i > 18:
            continue
        lemma = tok.lemma_.lower()
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", lemma):
            continue
        cands.append((tok, subj, lemma, subj_num))
    return cands


def _spacy_attractors_between(doc: Any, subj: Any, verb: Any, subject_number: str) -> tuple[list[int], list[str]]:
    idxs: list[int] = []
    nums: list[str] = []
    for tok in doc[subj.i + 1 : verb.i]:
        if tok.is_punct or tok.is_space:
            continue
        if tok.pos_ not in {"NOUN", "PROPN", "PRON"}:
            continue
        num = _spacy_subject_number(tok)
        if num is None:
            continue
        idxs.append(tok.i)
        nums.append(num)
    return idxs, nums


def _spacy_frame_shape(doc: Any, subj: Any, verb: Any) -> str:
    between = list(doc[subj.i + 1 : verb.i])
    pp_count = sum(1 for t in between if t.pos_ == "ADP" or t.text.lower() in PREPOSITIONS)
    has_rel = any(t.dep_ in {"relcl", "acl"} or t.text.lower() in REL_WORDS for t in between)
    comma = any(t.text in {",", ";", ":"} for t in doc)
    return f"pp{min(pp_count,2)}_rel{int(has_rel)}_comma{int(comma)}"


def extract_candidate_spacy_doc(doc: Any, source_id: str, split: str = "pool", min_len: int = 8, max_len: int = 96, hop_distance: int = 4) -> tuple[SourceSentence | None, Rejection | None]:
    sentence = doc.text.strip()
    noisy = _is_noisy_corpus_sentence(sentence)
    if noisy:
        return None, Rejection(sentence, noisy)
    if any(ch in sentence for ch in ['"', "“", "”", "<", ">", "|", "_"]):
        return None, Rejection(sentence, "quote_markup_or_underscore")
    tokens = _spacy_clean_tokens(doc)
    if len(tokens) < min_len or len(tokens) > max_len:
        return None, Rejection(sentence, "length_filter")
    if tokens and tokens[0].islower():
        return None, Rejection(sentence, "lowercase_sentence_initial")
    candidates = _spacy_candidate_verb_tokens(doc)
    if len(candidates) != 1:
        return None, Rejection(sentence, f"spacy_qualifying_verb_count_{len(candidates)}")
    verb, subj, lemma, subject_number = candidates[0]
    if sum(1 for tok in tokens[verb.i + 1 :] if is_word_token(tok)) < hop_distance:
        return None, Rejection(sentence, "not_enough_postverb_words")
    marker = "S" if subject_number == "singular" else "P"
    attractor_indices, attractor_numbers = _spacy_attractors_between(doc, subj, verb, subject_number)
    if len(attractor_numbers) == 0:
        template = "plain"
    elif any(n != subject_number for n in attractor_numbers):
        if any(n == subject_number for n in attractor_numbers):
            return None, Rejection(sentence, "mixed_same_and_opposite_attractors")
        template = "attractor_opposite"
    else:
        template = "attractor_same"
    frame = _spacy_frame_shape(doc, subj, verb)
    punct_cond, punct_count = _punctuation_in_window(tokens, verb.i, hop_distance)
    return SourceSentence(
        source_id=source_id,
        tokens=tokens,
        verb_index=verb.i,
        verb_inflected=verb.text,
        verb_lemma=lemma,
        marker=marker,
        subject_number=subject_number,
        split=split,
        template=template,
        has_attractor=bool(attractor_numbers),
        attractor_number=attractor_numbers[-1] if attractor_numbers else None,
        head_noun_index=subj.i,
        head_noun=subj.text,
        attractor_indices=attractor_indices,
        attractor_numbers=attractor_numbers,
        attractor_count=min(2, len(attractor_numbers)),
        frame_shape=frame,
        frame_split_type="seen_frame",
        punctuation_condition=punct_cond,
        window_punctuation_count=punct_count,
        tail_word_length=sum(1 for tok in tokens[verb.i + 1 :] if is_word_token(tok)),
        source_length=len(tokens),
        metadata={
            "parser": "spacy_dependency",
            "original_sentence": sentence,
            "spacy_model": getattr(doc.vocab, "lang", "unknown"),
            "subject_dep": subj.dep_,
            "subject_pos": subj.pos_,
            "subject_tag": subj.tag_,
            "verb_pos": verb.pos_,
            "verb_tag": verb.tag_,
        },
    ), None

def extract_candidate(sentence: str, source_id: str, split: str = "pool", min_len: int = 8, max_len: int = 64, hop_distance: int = 4) -> tuple[SourceSentence | None, Rejection | None]:
    noisy = _is_noisy_corpus_sentence(sentence)
    if noisy:
        return None, Rejection(sentence, noisy)
    if any(ch in sentence for ch in ['"', "“", "”", "<", ">", "|", "_"]):
        return None, Rejection(sentence, "quote_markup_or_underscore")
    tokens = tokenize_simple(sentence)
    if len(tokens) < min_len or len(tokens) > max_len:
        return None, Rejection(sentence, "length_filter")
    if tokens and tokens[0].islower():
        # Most accepted examples should be complete sentences, not transcript
        # action fragments like "pastes tape on paper".
        return None, Rejection(sentence, "lowercase_sentence_initial")
    lows = [t.lower() for t in tokens]
    qualifying: list[tuple[int, str, str, str]] = []
    for i, low in enumerate(lows):
        if low in INFL_SET:
            qualifying.append((i, INFL_TO_LEMMA[low], tokens[i], "singular"))
        elif low in LEMMA_SET:
            # Candidate plural present form; keep only if head noun is plural below.
            qualifying.append((i, low, tokens[i], "plural_candidate"))
    if len(qualifying) != 1:
        return None, Rejection(sentence, f"qualifying_verb_count_{len(qualifying)}")
    verb_index, lemma, inflected, verb_form_number = qualifying[0]
    if sum(1 for t in tokens[verb_index + 1 :] if is_word_token(t)) < hop_distance:
        return None, Rejection(sentence, "not_enough_postverb_words")
    head = _head_noun_before_verb(tokens, verb_index)
    if head is None:
        return None, Rejection(sentence, "no_clear_subject_head")
    head_idx, head_noun, head_number = head
    # Require the head to be reasonably close to the verb; long fragments are
    # often parser errors under the heuristic extractor.
    if verb_index - head_idx > 14:
        return None, Rejection(sentence, "subject_too_far_from_verb")
    if verb_form_number == "singular" and head_number != "singular":
        return None, Rejection(sentence, "verb_subject_number_mismatch")
    if verb_form_number == "plural_candidate" and head_number != "plural":
        return None, Rejection(sentence, "plural_candidate_without_plural_head")
    subject_number = head_number
    marker = "S" if subject_number == "singular" else "P"
    verb_lemma = lemma
    verb_inflected = inflected
    attractor_indices, attractor_numbers = _attractors_between(tokens, head_idx, verb_index, subject_number)
    if len(attractor_numbers) == 0:
        template = "plain"
    elif any(n != subject_number for n in attractor_numbers):
        if any(n == subject_number for n in attractor_numbers):
            return None, Rejection(sentence, "mixed_same_and_opposite_attractors")
        template = "attractor_opposite"
    else:
        template = "attractor_same"
    frame = _frame_shape(tokens, head_idx, verb_index)
    punct_cond, punct_count = _punctuation_in_window(tokens, verb_index, hop_distance)
    return SourceSentence(
        source_id=source_id,
        tokens=tokens,
        verb_index=verb_index,
        verb_inflected=verb_inflected,
        verb_lemma=verb_lemma,
        marker=marker,
        subject_number=subject_number,
        split=split,
        template=template,
        has_attractor=bool(attractor_numbers),
        attractor_number=attractor_numbers[-1] if attractor_numbers else None,
        head_noun_index=head_idx,
        head_noun=head_noun,
        attractor_indices=attractor_indices,
        attractor_numbers=attractor_numbers,
        attractor_count=min(2, len(attractor_numbers)),
        frame_shape=frame,
        frame_split_type="seen_frame",
        punctuation_condition=punct_cond,
        window_punctuation_count=punct_count,
        tail_word_length=sum(1 for tok in tokens[verb_index + 1 :] if is_word_token(tok)),
        source_length=len(tokens),
        metadata={"parser": "heuristic_v4_1d_strict_single_verb", "original_sentence": sentence},
    ), None


def make_demo_corpus(n: int, seed: int = 0) -> list[str]:
    """Generate natural-looking demo carriers for smoke tests.

    This is not the BabyLM scientific corpus. It deliberately creates matched
    singular/plural pairs with the same frame, verb index, length bucket, and
    tail distribution so the v4.1 shortcut gates can be exercised locally when
    no BabyLM files are available.
    """
    rng = random.Random(seed)
    sents: list[str] = []
    preps = ["of", "near", "beside", "behind", "around", "with", "inside"]
    # These are intentionally not in VERB_PAIRS, so the main-verb detector still
    # sees a single qualifying present-tense verb.
    rel_verbs = ["mentioned", "remembered", "noticed", "admired", "described"]
    filler = ["quietly", "outside", "again", "today", "nearby", "often", "afterward", "before", "sunset", "in", "the", "garden", "near", "the", "door", "with", "care"]

    def one(number: str, structure: dict) -> str:
        head = rng.choice(SINGULAR_HEADS if number == "singular" else PLURAL_HEADS)
        lemma, infl_s = structure["verb_pair"]
        verb = infl_s if number == "singular" else lemma
        subj = ["The"] + structure["adjs"] + [head]
        for attr_role in structure["attrs"]:
            # In opposite frames, every attractor is opposite; in same frames,
            # every attractor matches. This makes shortcut baselines auditable.
            if attr_role == "opposite":
                attr_num = "plural" if number == "singular" else "singular"
            elif attr_role == "same":
                attr_num = number
            else:
                attr_num = rng.choice(["singular", "plural"])
            attr = rng.choice(SINGULAR_ATTRACTORS if attr_num == "singular" else PLURAL_ATTRACTORS)
            subj += [rng.choice(preps), "the", rng.choice(ADJECTIVES), attr]
        if structure["rel"]:
            rel_num = "plural" if structure["rel_attr"] == "plural" else "singular"
            rel_noun = rng.choice(SINGULAR_ATTRACTORS if rel_num == "singular" else PLURAL_ATTRACTORS)
            subj += ["that", rng.choice(rel_verbs), "the", rel_noun]
        tail = list(structure["tail"])
        if structure["comma_in_window"]:
            tail = tail[:1] + [","] + tail[1:]
        tail = tail + list(structure["extra_tail"])
        return detokenize(subj + [verb] + tail + ["."])

    pairs = max(1, n // 2)
    for i in range(pairs):
        frame_type = i % 6
        if frame_type == 0:
            attrs, rel = [], False
        elif frame_type == 1:
            attrs, rel = ["same" if i % 4 == 0 else "opposite"], False
        elif frame_type == 2:
            attrs, rel = ["opposite", "opposite"], False
        elif frame_type == 3:
            attrs, rel = [], True
        elif frame_type == 4:
            attrs, rel = ["opposite"], True
        else:
            attrs, rel = ["same", "same"], True
        structure = {
            "verb_pair": rng.choice(VERB_PAIRS),
            "adjs": rng.sample(ADJECTIVES, rng.randint(0, 5)),
            "attrs": attrs,
            "rel": rel,
            "rel_attr": rng.choice(["singular", "plural"]),
            "tail": rng.choice(TAIL_PHRASES),
            "comma_in_window": (i % 3 == 0),
            "extra_tail": [rng.choice(filler) for _ in range(rng.randint(0, 30))],
        }
        sents.append(one("singular", structure))
        sents.append(one("plural", structure))
    rng.shuffle(sents)
    return sents[:n]

def _normal_hash(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()

def _local_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy) ** 0.5


def _source_marker_index(c: SourceSentence, hop_distance: int = 4) -> int:
    """Fast WORDHOP insertion index without materialising the transformed text."""
    count_words = 0
    last_word_idx = c.verb_index
    for idx in range(c.verb_index + 1, len(c.tokens)):
        if is_word_token(c.tokens[idx]):
            count_words += 1
            last_word_idx = idx
            if count_words >= hop_distance:
                return idx + 1
    return min(len(c.tokens), last_word_idx + 1)


def _source_tail_after_marker(c: SourceSentence, hop_distance: int = 4) -> int:
    marker_idx = _source_marker_index(c, hop_distance)
    return sum(1 for tok in c.tokens[marker_idx:] if is_word_token(tok))


def _subset_position_metrics(sources: list[SourceSentence]) -> dict[str, float]:
    if not sources:
        return {
            "mode_marker_share": 1.0,
            "mode_verb_share": 1.0,
            "marker_length_corr": 1.0,
            "value_verb_corr": 1.0,
            "value_length_corr": 1.0,
            "tail_verb_corr": 1.0,
        }
    marker_idx = [float(_source_marker_index(c)) for c in sources]
    lengths = [float(c.source_length) for c in sources]
    verb_idx = [float(c.verb_index) for c in sources]
    values = [1.0 if c.marker == "P" else 0.0 for c in sources]
    tails = [float(_source_tail_after_marker(c)) for c in sources]
    return {
        "mode_marker_share": max(Counter(int(x) for x in marker_idx).values()) / len(marker_idx),
        "mode_verb_share": max(Counter(int(x) for x in verb_idx).values()) / len(verb_idx),
        "marker_length_corr": abs(_local_corr(marker_idx, lengths)),
        "value_verb_corr": abs(_local_corr(values, verb_idx)),
        "value_length_corr": abs(_local_corr(values, lengths)),
        "tail_verb_corr": abs(_local_corr(tails, verb_idx)),
    }


def _subset_leak_score(sources: list[SourceSentence]) -> float:
    """Score a split by the shortcut signals the v4.1 gate rejects.

    The score deliberately puts a large penalty on the two failure modes seen on
    the first BabyLM run: a high WORDHOP mode-slot baseline and a high
    marker-index/length correlation. This does not change labels; it only
    chooses a less shortcut-prone held-out sample from the same corpus pool.
    """
    m = _subset_position_metrics(sources)
    # Smooth penalties below the hard gate, steep penalties above it.
    mode_excess = max(0.0, m["mode_marker_share"] - 0.145)
    corr_excess = max(0.0, m["marker_length_corr"] - 0.28)
    tail_excess = max(0.0, m["tail_verb_corr"] - 0.19)
    # v4.1d: the only remaining BabyLM failure after v4.1c was the
    # WORDHOP C4 tail/verb correlation. This diagnostic matters because a
    # fixed relation between where the verb appears and how much text remains
    # after the marker can make placement partially length-anchored. Give it a
    # hard excess penalty, while still preserving the earlier placement/value
    # shortcut objectives.
    return (
        4.0 * m["marker_length_corr"]
        + 2.0 * m["value_verb_corr"]
        + 2.0 * m["value_length_corr"]
        + 6.0 * m["tail_verb_corr"]
        + 3.0 * m["mode_marker_share"]
        + 1.0 * m["mode_verb_share"]
        + 60.0 * mode_excess
        + 25.0 * corr_excess
        + 120.0 * tail_excess
    )


def _marker_counts(cands: list[SourceSentence]) -> Counter[str]:
    return Counter(c.subject_number for c in cands)


def _can_balance(cands: list[SourceSentence], n: int) -> bool:
    cnt = _marker_counts(cands)
    return cnt["singular"] >= n // 2 and cnt["plural"] >= n - n // 2


def _draw_diverse(pool: list[SourceSentence], k: int, rng: random.Random) -> list[SourceSentence]:
    """Draw k examples while spreading verb positions and lengths."""
    if k <= 0:
        return []
    by_key: defaultdict[tuple[int, int, str], list[SourceSentence]] = defaultdict(list)
    for c in pool:
        # Use the actual WORDHOP marker slot plus coarse length/tail buckets;
        # this directly attacks mode-slot and length-anchored shortcuts while
        # still preserving real-corpus sentence variation.
        key = (_source_marker_index(c), c.source_length // 4, _source_tail_after_marker(c) // 4, c.frame_shape)
        by_key[key].append(c)
    for vals in by_key.values():
        rng.shuffle(vals)
    keys = list(by_key)
    rng.shuffle(keys)
    chosen: list[SourceSentence] = []
    while len(chosen) < k and keys:
        rng.shuffle(keys)
        next_keys = []
        for key in keys:
            vals = by_key[key]
            if vals and len(chosen) < k:
                chosen.append(vals.pop())
            if vals:
                next_keys.append(key)
        keys = next_keys
    if len(chosen) < k:
        rest = [c for vals in by_key.values() for c in vals]
        rng.shuffle(rest)
        chosen.extend(rest[: k - len(chosen)])
    return chosen[:k]


def _choose_balanced(cands: list[SourceSentence], n: int, rng: random.Random) -> list[SourceSentence]:
    """Choose an exactly S/P-balanced subset with diverse positions.

    If exact balance is impossible, return fewer than n examples; the caller
    should choose a different split or fail loudly. Silent all-S/all-P probes
    are invalid for the v4.1 value gate.
    """
    if not _can_balance(cands, n):
        return []
    target_s = n // 2
    target_p = n - target_s
    singular = [c for c in cands if c.subject_number == "singular"]
    plural = [c for c in cands if c.subject_number == "plural"]

    # First, take matched S/P pairs inside the same position/length/frame bucket.
    buckets: defaultdict[tuple[int, int, str], dict[str, list[SourceSentence]]] = defaultdict(lambda: {"singular": [], "plural": []})
    for c in cands:
        # Pair S/P examples within a bucket that includes the real WORDHOP
        # marker slot. This prevents value from becoming recoverable from
        # subject-region length or marker position.
        key = (_source_marker_index(c) // 2, c.source_length // 4, _source_tail_after_marker(c) // 4, c.frame_shape)
        buckets[key][c.subject_number].append(c)
    for b in buckets.values():
        rng.shuffle(b["singular"])
        rng.shuffle(b["plural"])

    chosen: list[SourceSentence] = []
    used: set[str] = set()
    keys = list(buckets)
    rng.shuffle(keys)
    cur_s = cur_p = 0
    made_progress = True
    while made_progress and cur_s < target_s and cur_p < target_p:
        made_progress = False
        rng.shuffle(keys)
        for key in keys:
            if cur_s >= target_s or cur_p >= target_p:
                break
            b = buckets[key]
            if b["singular"] and b["plural"]:
                s = b["singular"].pop()
                p = b["plural"].pop()
                chosen.extend([s, p])
                used.add(s.source_id); used.add(p.source_id)
                cur_s += 1; cur_p += 1
                made_progress = True

    # Fill each side separately with diverse draws.
    rem_s = [c for c in singular if c.source_id not in used]
    rem_p = [c for c in plural if c.source_id not in used]
    chosen.extend(_draw_diverse(rem_s, target_s - cur_s, rng))
    chosen.extend(_draw_diverse(rem_p, target_p - cur_p, rng))
    if len({c.source_id for c in chosen}) < n:
        # Remove accidental duplicates and top up from unused pools.
        uniq = []
        seen = set()
        for c in chosen:
            if c.source_id not in seen:
                uniq.append(c); seen.add(c.source_id)
        chosen = uniq
        for pool in (rem_s, rem_p):
            for c in pool:
                if len(chosen) >= n:
                    break
                if c.source_id not in seen:
                    chosen.append(c); seen.add(c.source_id)
    if len(chosen) != n:
        return []
    rng.shuffle(chosen)
    return chosen


def _choose_balanced_low_leak(cands: list[SourceSentence], n: int, rng: random.Random, trials: int = 260) -> list[SourceSentence]:
    best: list[SourceSentence] | None = None
    best_score = 999.0
    if not _can_balance(cands, n):
        return []
    for _ in range(trials):
        cc = list(cands)
        rng.shuffle(cc)
        chosen = _choose_balanced(cc, n, rng)
        if len(chosen) < n:
            continue
        score = _subset_leak_score(chosen)
        if score < best_score:
            best, best_score = chosen, score
    return best if best is not None else []




def _choose_probe_with_heldout(
    train_pool: list[SourceSentence],
    heldout_pool: list[SourceSentence],
    n_probe: int,
    rng: random.Random,
    min_held: int,
    trials: int = 260,
) -> tuple[list[SourceSentence], list[SourceSentence]]:
    """Choose a balanced probe split while explicitly minimising v4.1 leaks.

    The train/probe split is not semantically special for HOP; what matters is
    that the probe is held out and cannot be solved by position-only rules. The
    first v4.1b BabyLM run failed only on probe shortcut gates, so v4.1c gives
    the probe first priority, then draws train from the remaining source pool.
    """
    if not _can_balance(train_pool, n_probe - min_held):
        return [], []
    best_probe: list[SourceSentence] | None = None
    best_held: list[SourceSentence] = []
    best_score = 999999.0
    held_options = [min_held]
    if min_held >= 20:
        held_options += [max(0, min_held - 10), min(n_probe // 3, min_held + 10)]
    held_options = sorted(set(h for h in held_options if 0 <= h <= n_probe))
    for _ in range(trials):
        rng.shuffle(train_pool)
        rng.shuffle(heldout_pool)
        for n_held in held_options:
            held = []
            if n_held:
                if not _can_balance(heldout_pool, n_held):
                    continue
                held = _choose_balanced_low_leak(heldout_pool, n_held, rng, trials=4)
                if len(held) != n_held:
                    continue
            used = {c.source_id for c in held}
            seen_needed = n_probe - len(held)
            seen_pool = [c for c in train_pool if c.source_id not in used]
            if not _can_balance(seen_pool, seen_needed):
                continue
            seen = _choose_balanced_low_leak(seen_pool, seen_needed, rng, trials=6)
            if len(seen) != seen_needed:
                continue
            probe = held + seen
            if len({c.source_id for c in probe}) != n_probe or not _can_balance(probe, n_probe):
                continue
            score = _subset_leak_score(probe)
            # Slightly prefer retaining a true held-out-frame slice when scores tie.
            score -= 0.03 * len(held) / max(1, n_probe)
            if score < best_score:
                best_probe = probe
                best_held = held
                best_score = score
    if best_probe is None:
        return [], []
    rng.shuffle(best_probe)
    return best_probe, best_held

def build_corpus_hop_dataset(
    corpus_globs: list[str] | None = None,
    n_train: int = 1000,
    n_probe: int = 300,
    seed: int = 0,
    hop_distance: int = 4,
    include_tokenhop: bool = False,
    use_demo_if_no_corpus: bool = True,
    max_corpus_sentences: int | None = None,
    parser_backend: str = "heuristic",
    spacy_model: str = "en_core_web_sm",
    spacy_batch_size: int = 128,
    require_parser: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = random.Random(seed)
    corpus_globs = corpus_globs or []
    raw_sentences = load_corpus_sentences(corpus_globs, limit=max_corpus_sentences)
    source_kind = "corpus"
    if not raw_sentences and use_demo_if_no_corpus:
        raw_sentences = make_demo_corpus(max(6000, (n_train + n_probe) * 10), seed=seed + 17)
        source_kind = "demo_fallback"
    rejections: list[Rejection] = []
    rejection_counter: Counter[str] = Counter()
    seen = set()
    cands: list[SourceSentence] = []
    parser_backend = str(parser_backend or "heuristic").lower()
    if parser_backend not in {"heuristic", "spacy"}:
        raise ValueError(f"Unknown parser_backend={parser_backend!r}; expected 'heuristic' or 'spacy'")

    indexed_sentences: list[tuple[int, str]] = []
    for i, sent in enumerate(raw_sentences):
        norm = _normal_hash(sent)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        indexed_sentences.append((i, sent))

    if parser_backend == "spacy":
        try:
            import spacy  # type: ignore
            try:
                nlp = spacy.load(spacy_model, disable=["ner"])
            except OSError as e:
                raise RuntimeError(
                    f"spaCy model {spacy_model!r} is not installed. Install it with: "
                    f"python -m spacy download {spacy_model}"
                ) from e
        except Exception:
            if require_parser:
                raise
            # Fall back only when explicitly allowed; the report marks this.
            nlp = None
            parser_backend = "heuristic_fallback_after_spacy_unavailable"
        if parser_backend == "spacy":
            texts = [s for _, s in indexed_sentences]
            ids = [i for i, _ in indexed_sentences]
            for raw_i, doc in zip(ids, nlp.pipe(texts, batch_size=spacy_batch_size)):
                cand, rej = extract_candidate_spacy_doc(doc, f"corpus_{raw_i:07d}", hop_distance=hop_distance)
                if cand is not None:
                    cands.append(cand)
                elif rej is not None:
                    rejection_counter[rej.reason] += 1
                    if len(rejections) < 500:
                        rejections.append(rej)
    if parser_backend != "spacy":
        for i, sent in indexed_sentences:
            cand, rej = extract_candidate(sent, f"corpus_{i:07d}", hop_distance=hop_distance)
            if cand is not None:
                cands.append(cand)
            elif rej is not None:
                rejection_counter[rej.reason] += 1
                if len(rejections) < 500:
                    rejections.append(rej)

    need_total = n_train + n_probe
    counts_all = _marker_counts(cands)
    if len(cands) < need_total or counts_all["singular"] < need_total // 2 or counts_all["plural"] < need_total - need_total // 2:
        raise ValueError(
            "Insufficient balanced parse-valid corpus candidates. "
            f"found total={len(cands)}, singular={counts_all['singular']}, plural={counts_all['plural']}; "
            f"need at least total={need_total} with both values. "
            "Use more corpus sentences, remove transcript-heavy files, or broaden the parser."
        )

    by_frame: defaultdict[str, list[SourceSentence]] = defaultdict(list)
    for c in cands:
        by_frame[c.frame_shape].append(c)

    # Choose a held-out frame that is non-dominant, has both values, and leaves
    # enough balanced candidates for training. Never let this frame leak into train.
    min_held = max(20, n_probe // 10)
    heldout_frame = None
    frame_candidates = sorted(
        by_frame,
        key=lambda f: (
            # prefer more complex frames, then enough but not dominant counts
            not ("pp2" in f or "rel1" in f),
            -min(_marker_counts(by_frame[f])["singular"], _marker_counts(by_frame[f])["plural"]),
            len(by_frame[f]),
        ),
    )
    for f in frame_candidates:
        held_counts = _marker_counts(by_frame[f])
        if held_counts["singular"] < min_held // 2 or held_counts["plural"] < min_held - min_held // 2:
            continue
        train_pool_try = [c for c in cands if c.frame_shape != f]
        if _can_balance(train_pool_try, n_train):
            # Enough seen-frame material must remain for the rest of the probe.
            seen_probe_need = n_probe - min_held
            if _can_balance(train_pool_try, n_train + seen_probe_need):
                heldout_frame = f
                break
    if heldout_frame is None:
        # Fall back to a frame with both values and enough train material. The
        # validator will still fail if no true held-out frame is possible.
        for f in frame_candidates:
            train_pool_try = [c for c in cands if c.frame_shape != f]
            if _can_balance(by_frame[f], min_held) and _can_balance(train_pool_try, n_train):
                heldout_frame = f
                break
    if heldout_frame is None:
        raise ValueError(
            "Could not choose a balanced held-out frame without starving train. "
            f"Frame counts: {dict(Counter(c.frame_shape for c in cands))}"
        )

    train_pool = [c for c in cands if c.frame_shape != heldout_frame]
    heldout_pool = [c for c in cands if c.frame_shape == heldout_frame]
    rng.shuffle(train_pool); rng.shuffle(heldout_pool)

    # v4.1c: choose the held-out probe first because the hard scientific gate
    # is defined on the probe distribution. Then draw train from remaining
    # non-heldout sources. This avoids consuming the rare anti-correlated corpus
    # examples in train and leaving a shortcut-prone probe.
    n_held_min = min(max(20, n_probe // 10), n_probe // 3)
    if not _can_balance(heldout_pool, n_held_min):
        n_held_min = 0
    probe_base, heldout_chosen = _choose_probe_with_heldout(
        train_pool=train_pool,
        heldout_pool=heldout_pool,
        n_probe=n_probe,
        rng=rng,
        min_held=n_held_min,
        trials=360,
    )
    if len(probe_base) != n_probe or not _can_balance(probe_base, n_probe):
        raise ValueError(
            f"Could not draw low-leak balanced probe split. "
            f"heldout_frame={heldout_frame}, train_pool_counts={dict(_marker_counts(train_pool))}, "
            f"heldout_counts={dict(_marker_counts(heldout_pool))}"
        )
    probe_ids = {c.source_id for c in probe_base}
    train_candidates = [c for c in train_pool if c.source_id not in probe_ids]
    train_base = _choose_balanced_low_leak(train_candidates, n_train, rng, trials=360)
    if len(train_base) != n_train:
        raise ValueError(
            f"Could not draw balanced train split from non-heldout/non-probe frames. "
            f"heldout_frame={heldout_frame}, train_counts={dict(_marker_counts(train_candidates))}"
        )
    rng.shuffle(probe_base)

    def update(c: SourceSentence, split: str) -> SourceSentence:
        import dataclasses
        return dataclasses.replace(
            c,
            split=split,
            frame_split_type="heldout_frame" if (split == "probe" and c.frame_shape == heldout_frame) else "seen_frame",
            metadata={**c.metadata, "source_kind": source_kind, "heldout_frame": heldout_frame},
        )
    train_sources = [update(c, "train") for c in train_base]
    probe_sources = [update(c, "probe") for c in probe_base]

    arms = ["NOHOP", "WORDHOP"] + (["TOKENHOP"] if include_tokenhop else [])
    data: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        key = arm.lower()
        data[f"{key}_train"] = [transform_sentence(c, arm, hop_distance) for c in train_sources]
        data[f"{key}_probe"] = [transform_sentence(c, arm, hop_distance) for c in probe_sources]

    meta = {
        "source_kind": source_kind,
        "n_raw_sentences": len(raw_sentences),
        "n_parse_valid_candidates": len(cands),
        "n_rejections_sampled": len(rejections),
        "heldout_frame": heldout_frame,
        "candidate_marker_counts": dict(counts_all),
        "train_marker_counts": dict(_marker_counts(train_sources)),
        "probe_marker_counts": dict(_marker_counts(probe_sources)),
        "probe_position_metrics_pre_transform": _subset_position_metrics(probe_sources),
        "train_position_metrics_pre_transform": _subset_position_metrics(train_sources),
        "frame_counts_valid": dict(Counter(c.frame_shape for c in cands)),
        "frame_marker_counts_valid": {
            f: dict(_marker_counts(vals)) for f, vals in by_frame.items()
        },
        "rejection_reasons_total": dict(rejection_counter),
        "rejection_reasons_sampled": dict(Counter(r.reason for r in rejections)),
        "rejection_examples": [{"reason": r.reason, "sentence": r.sentence} for r in rejections[:50]],
        "parser": parser_backend,
        "spacy_model": spacy_model if parser_backend == "spacy" else None,
        "require_parser": require_parser,
        "single_qualifying_verb_policy": True,
    }
    return data, meta


def unigram_loss(texts: Iterable[str], ref_counter: Counter[str], vocab_size: int) -> float:
    total = 0.0
    n = 0
    denom = sum(ref_counter.values()) + vocab_size
    for text in texts:
        for tok in tokenize_simple(text.lower()):
            if not is_word_token(tok):
                continue
            p = (ref_counter[tok] + 1) / denom
            total += -math.log(p)
            n += 1
    return total / max(1, n)


def naturalness_summary(carrier_texts: list[str], reference_texts: list[str], threshold: float = 1.5) -> tuple[list[str], dict[str, Any]]:
    # Lightweight proxy for the v4.1 base-model naturalness gate. For the real
    # BabyLM/Pythia run, this should be supplemented by a base-model loss pass.
    ref_counter: Counter[str] = Counter()
    for text in reference_texts:
        for tok in tokenize_simple(text.lower()):
            if is_word_token(tok):
                ref_counter[tok] += 1
    vocab_size = max(1, len(ref_counter))
    carrier_loss = unigram_loss(carrier_texts, ref_counter, vocab_size)
    ref_loss = unigram_loss(reference_texts, ref_counter, vocab_size)
    ratio = carrier_loss / max(ref_loss, 1e-9)
    errors = []
    if ratio > threshold:
        errors.append(f"carrier naturalness proxy ratio too high: {ratio:.3f} > {threshold}")
    return errors, {
        "proxy": "add-one unigram loss on unmarked carriers against corpus reference; replace/supplement with base-LM loss for final gate",
        "carrier_mean_loss": carrier_loss,
        "reference_mean_loss": ref_loss,
        "carrier_reference_loss_ratio": ratio,
        "pass_threshold": threshold,
        "n_carriers": len(carrier_texts),
        "n_reference": len(reference_texts),
    }
