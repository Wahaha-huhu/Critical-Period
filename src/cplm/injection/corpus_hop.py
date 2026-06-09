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


def _number_of_noun(tok: str) -> str | None:
    t = tok.lower()
    if t in SINGULAR_NOUNS:
        return "singular"
    if t in PLURAL_NOUNS:
        return "plural"
    if t in {"children", "people", "men", "women"}:
        return "plural"
    if len(t) <= 2 or t in FUNCTION or t in LEMMA_SET or t in INFL_SET:
        return None
    # Conservative morphology fallback. Avoid many false plural endings.
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return "plural"
    if re.fullmatch(r"[a-z]+", t):
        return "singular"
    return None


def _head_noun_before_verb(tokens: list[str], verb_index: int) -> tuple[int, str, str] | None:
    # Choose the first plausible noun in the pre-verbal subject region. This
    # deliberately avoids nearest-noun behavior in PP attractor cases.
    for i, tok in enumerate(tokens[:verb_index]):
        low = tok.lower()
        if low in PUNCT or low in FUNCTION or low in ADJECTIVES:
            continue
        num = _number_of_noun(tok)
        if num:
            return i, tok, num
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


def extract_candidate(sentence: str, source_id: str, split: str = "pool", min_len: int = 8, max_len: int = 64, hop_distance: int = 4) -> tuple[SourceSentence | None, Rejection | None]:
    if any(ch in sentence for ch in ['"', "“", "”", "<", ">", "|"]):
        return None, Rejection(sentence, "quote_or_markup")
    tokens = tokenize_simple(sentence)
    if len(tokens) < min_len or len(tokens) > max_len:
        return None, Rejection(sentence, "length_filter")
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
    if verb_form_number == "singular" and head_number != "singular":
        return None, Rejection(sentence, "verb_subject_number_mismatch")
    if verb_form_number == "plural_candidate" and head_number != "plural":
        return None, Rejection(sentence, "plural_candidate_without_plural_head")
    subject_number = head_number
    marker = "S" if subject_number == "singular" else "P"
    # For plural candidate the inflected token is already the lemma. For singular,
    # ensure the lemma is known.
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
        metadata={"parser": "heuristic_v4_1", "original_sentence": sentence},
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


def _subset_leak_score(sources: list[SourceSentence]) -> float:
    if not sources:
        return 999.0
    recs = [transform_sentence(c, "WORDHOP", 4) for c in sources]
    marker_idx = [float(r["marker_index"]) for r in recs]
    lengths = [float(r["source_length"]) for r in recs]
    verb_idx = [float(r["verb_index_transformed"]) for r in recs]
    values = [1.0 if r["marker"] == "P" else 0.0 for r in recs]
    mode_share = max(Counter(int(x) for x in marker_idx).values()) / len(marker_idx)
    return abs(_local_corr(marker_idx, lengths)) + abs(_local_corr(values, verb_idx)) + 0.5 * mode_share


def _choose_balanced_low_leak(cands: list[SourceSentence], n: int, rng: random.Random, trials: int = 80) -> list[SourceSentence]:
    best: list[SourceSentence] | None = None
    best_score = 999.0
    for _ in range(trials):
        # Shuffle copy so _choose_balanced sees different within-bucket order.
        cc = list(cands)
        rng.shuffle(cc)
        chosen = _choose_balanced(cc, n, rng)
        if len(chosen) < n:
            continue
        score = _subset_leak_score(chosen)
        if score < best_score:
            best, best_score = chosen, score
    return best if best is not None else _choose_balanced(cands, n, rng)


def _choose_balanced(cands: list[SourceSentence], n: int, rng: random.Random) -> list[SourceSentence]:
    """Choose a roughly S/P-balanced subset while decorrelating value from position.

    We greedily draw matched singular/plural examples inside coarse buckets of
    verb index, source length, and frame. This implements the v4.1 value-side
    rebalancing before the validator runs the value-from-position attack.
    """
    buckets: defaultdict[tuple[int, int, str], dict[str, list[SourceSentence]]] = defaultdict(lambda: {"singular": [], "plural": []})
    for c in cands:
        key = (c.verb_index // 2, c.source_length // 4, c.frame_shape)
        buckets[key][c.subject_number].append(c)
    chosen: list[SourceSentence] = []
    keys = list(buckets)
    rng.shuffle(keys)
    # First take within-bucket pairs where possible.
    made_progress = True
    while len(chosen) + 2 <= n and made_progress:
        made_progress = False
        rng.shuffle(keys)
        for key in keys:
            if len(chosen) + 2 > n:
                break
            b = buckets[key]
            if b["singular"] and b["plural"]:
                chosen.append(b["singular"].pop())
                chosen.append(b["plural"].pop())
                made_progress = True
    # Fill remaining slots with global balance.
    existing = {c.source_id for c in chosen}
    rest_by = {"singular": [], "plural": []}
    for c in cands:
        if c.source_id not in existing:
            rest_by[c.subject_number].append(c)
    rng.shuffle(rest_by["singular"]); rng.shuffle(rest_by["plural"])
    target_s = n // 2
    target_p = n - target_s
    cur_s = sum(1 for c in chosen if c.subject_number == "singular")
    cur_p = len(chosen) - cur_s
    while len(chosen) < n and (rest_by["singular"] or rest_by["plural"]):
        need_s = cur_s < target_s
        pick_num = "singular" if need_s and rest_by["singular"] else "plural" if rest_by["plural"] else "singular"
        c = rest_by[pick_num].pop()
        chosen.append(c)
        cur_s += int(pick_num == "singular")
        cur_p += int(pick_num == "plural")
    rng.shuffle(chosen)
    return chosen[:n]


def build_corpus_hop_dataset(
    corpus_globs: list[str] | None = None,
    n_train: int = 1000,
    n_probe: int = 300,
    seed: int = 0,
    hop_distance: int = 4,
    include_tokenhop: bool = False,
    use_demo_if_no_corpus: bool = True,
    max_corpus_sentences: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = random.Random(seed)
    corpus_globs = corpus_globs or []
    raw_sentences = load_corpus_sentences(corpus_globs, limit=max_corpus_sentences)
    source_kind = "corpus"
    if not raw_sentences and use_demo_if_no_corpus:
        # Oversample because heuristic extraction rejects some generated lines with relative verbs.
        raw_sentences = make_demo_corpus(max(6000, (n_train + n_probe) * 8), seed=seed + 17)
        source_kind = "demo_fallback"
    rejections: list[Rejection] = []
    seen = set()
    cands: list[SourceSentence] = []
    for i, sent in enumerate(raw_sentences):
        norm = _normal_hash(sent)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cand, rej = extract_candidate(sent, f"corpus_{i:07d}", hop_distance=hop_distance)
        if cand is not None:
            cands.append(cand)
        elif rej is not None and len(rejections) < 500:
            rejections.append(rej)

    if len(cands) < n_train + n_probe:
        raise ValueError(f"Only {len(cands)} parse-valid corpus candidates; need at least {n_train+n_probe}. Use a larger corpus or relax filters.")

    # Hold out the rarest available complex frame with enough examples; prefer pp2+relative if present.
    by_frame = defaultdict(list)
    for c in cands:
        by_frame[c.frame_shape].append(c)
    preferred = sorted(by_frame, key=lambda f: (not ("pp2" in f and "rel1" in f), -len(by_frame[f])))
    heldout_frame = None
    for f in preferred:
        if len(by_frame[f]) >= max(20, n_probe // 8):
            heldout_frame = f
            break
    if heldout_frame is None:
        heldout_frame = max(by_frame, key=lambda f: len(by_frame[f]))

    train_pool = [c for c in cands if c.frame_shape != heldout_frame]
    heldout_pool = [c for c in cands if c.frame_shape == heldout_frame]
    rng.shuffle(train_pool)
    rng.shuffle(heldout_pool)
    n_heldout_probe = min(len(heldout_pool), max(20, n_probe // 4))
    probe_seen_pool = train_pool[n_train:]
    train_candidates = train_pool[:max(n_train * 5, n_train)]
    if len(train_candidates) < n_train:
        rng.shuffle(cands)
        train_candidates = cands[:max(n_train * 5, n_train)]
    train_base = _choose_balanced_low_leak(train_candidates, n_train, rng, trials=60)
    train_ids = {c.source_id for c in train_base}

    heldout_candidates = [c for c in heldout_pool if c.source_id not in train_ids]
    seen_candidates = [c for c in train_pool if c.source_id not in train_ids]
    # Force a modest held-out-frame subset for the structural-generalisation
    # check, but keep it small enough that it does not dominate correlations.
    n_held = min(len(heldout_candidates), max(20, n_probe // 10))
    heldout_chosen = _choose_balanced_low_leak(heldout_candidates, n_held, rng, trials=40) if n_held else []
    seen_needed = n_probe - len(heldout_chosen)
    seen_chosen = _choose_balanced_low_leak(seen_candidates, seen_needed, rng, trials=120)
    probe_base = heldout_chosen + seen_chosen
    if len(probe_base) < n_probe:
        remaining = [c for c in cands if c.source_id not in train_ids and c.source_id not in {p.source_id for p in probe_base}]
        probe_base += _choose_balanced_low_leak(remaining, n_probe - len(probe_base), rng, trials=20)
    rng.shuffle(probe_base)

    # Set split and held-out metadata on immutable SourceSentence via replacement.
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
        "frame_counts_valid": dict(Counter(c.frame_shape for c in cands)),
        "rejection_reasons_sampled": dict(Counter(r.reason for r in rejections)),
        "rejection_examples": [{"reason": r.reason, "sentence": r.sentence} for r in rejections[:50]],
        "parser": "heuristic_v4_1_single_verb",
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
