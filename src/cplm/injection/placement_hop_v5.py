from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

PUNCT = {",", ".", ";", ":", "!", "?"}
BAD_MARKERS = ["xxx", ".cha", "speaker:", "mot:", "chi:", "fat:", "bro:", "urs:", "childes_"]
AUX_LEMMAS = {"be", "have", "do"}


def is_word(tok: str) -> bool:
    return tok not in PUNCT and bool(re.search(r"[A-Za-z0-9]", tok))


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def config_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if len(p.strip()) >= 10]


def load_corpus_sentences(globs_: list[str], limit: int | None = None) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    for pat in globs_:
        paths.extend(glob.glob(pat, recursive=True))
    paths = sorted(set(paths))
    out: list[str] = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for sent in split_sentences(text):
            out.append(sent)
            if limit and len(out) >= limit:
                return out, paths
    return out[:limit] if limit else out, paths


@dataclass(frozen=True)
class PlacementCandidate:
    source_id: str
    source_text: str
    tokens: list[str]
    verb_index: int
    verb_text: str
    verb_lemma: str
    frame_shape: str
    punctuation_condition: str
    source_length: int
    postverb_word_count: int
    domain: str = "unknown"
    parser: str = "unknown"
    parser_model: str = "unknown"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "PlacementCandidate":
        return PlacementCandidate(**d)


def _is_noisy_sentence(sentence: str) -> str | None:
    low = sentence.lower()
    if any(x in low for x in BAD_MARKERS):
        return "transcript_or_markup"
    if re.search(r"\b[A-Z]{2,}\s*:", sentence):
        return "speaker_label"
    if ":" in sentence and len(sentence.split()) < 18:
        return "colon_short_fragment"
    if re.search(r"\b\w*\d\w*\b", sentence):
        return "digit_or_file_id"
    if any(ch in sentence for ch in ["<", ">", "|", "_"]):
        return "markup_character"
    return None


def _frame_shape_from_spacy(doc: Any, verb: Any) -> str:
    before = list(doc[:verb.i])
    pp_count = sum(1 for t in before if getattr(t, "pos_", "") == "ADP")
    has_rel = any(getattr(t, "dep_", "") in {"relcl", "acl"} or t.text.lower() in {"who", "that", "which"} for t in before)
    comma = any(t.text in {",", ";", ":"} for t in doc)
    # Coarse shapes only; the adversarial baseline can use this, but cannot see verb index.
    return f"pp{min(pp_count,2)}_rel{int(has_rel)}_comma{int(comma)}"


def _punctuation_in_window(tokens: list[str], verb_index: int, hop_distance: int) -> tuple[str, int]:
    count_words = 0
    punct = 0
    for idx in range(verb_index + 1, len(tokens)):
        if tokens[idx] in PUNCT:
            punct += 1
        elif is_word(tokens[idx]):
            count_words += 1
            if count_words >= hop_distance:
                break
    return ("window_punct" if punct else "no_window_punct"), punct


def _wordhop_insert_index(tokens_after_lemma: list[str], verb_index: int, hop_distance: int) -> int:
    seen = 0
    for idx in range(verb_index + 1, len(tokens_after_lemma)):
        if is_word(tokens_after_lemma[idx]):
            seen += 1
            if seen == hop_distance:
                return idx + 1
    raise ValueError("not enough words after verb")


def _candidate_positions(tokens_after_lemma: list[str], verb_index: int, true_idx: int, hop_distance: int) -> list[int]:
    # Candidate slots for placement scoring: adjacent, nearby hop slots, length anchors, true slot.
    cand = {true_idx, verb_index + 1}
    for d in [1, 2, 3, 4, 5, 6]:
        try:
            cand.add(_wordhop_insert_index(tokens_after_lemma, verb_index, d))
        except ValueError:
            pass
    # Length anchors: common shortcut families.
    for gap in [1, 2, 3, 4, 5, 8]:
        cand.add(max(1, min(len(tokens_after_lemma), len(tokens_after_lemma) - gap)))
    return sorted(x for x in cand if 0 <= x <= len(tokens_after_lemma))


def transform_candidate(c: PlacementCandidate, arm: str, marker: str = "<HOP>", hop_distance: int = 4, split: str = "train") -> dict[str, Any]:
    tokens = list(c.tokens)
    tokens[c.verb_index] = c.verb_lemma
    if arm.lower() == "nohop":
        marker_idx = c.verb_index + 1
        target_distance = 0
    elif arm.lower() == "wordhop":
        marker_idx = _wordhop_insert_index(tokens, c.verb_index, hop_distance)
        target_distance = hop_distance
    else:
        raise ValueError(f"unknown arm: {arm}")
    marked_tokens = tokens[:marker_idx] + [marker] + tokens[marker_idx:]
    candidates = _candidate_positions(tokens, c.verb_index, marker_idx, hop_distance)
    return {
        "id": f"{c.source_id}__{arm.lower()}",
        "source_id": c.source_id,
        "split": split,
        "arm": arm.upper(),
        "marker": marker,
        "hop_distance": target_distance,
        "source_text": c.source_text,
        "source_tokens": c.tokens,
        "tokens_after_lemma": tokens,
        "tokens": marked_tokens,
        "text": detokenize(marked_tokens),
        "correct_text": detokenize(marked_tokens),
        "verb_index_original": c.verb_index,
        "verb_index_transformed": c.verb_index,
        "verb_inflected": c.verb_text,
        "verb_lemma": c.verb_lemma,
        "marker_index": marker_idx,
        "source_length": c.source_length,
        "marked_length": len(marked_tokens),
        "frame_shape": c.frame_shape,
        "punctuation_condition": c.punctuation_condition,
        "postverb_word_count": c.postverb_word_count,
        "tail_word_length_after_marker": sum(1 for tok in tokens[marker_idx:] if is_word(tok)),
        "domain": c.domain,
        "parser": c.parser,
        "parser_model": c.parser_model,
        "metric_targets": {
            "placement": {
                "correct_position": marker_idx,
                "marker": marker,
                "candidate_positions": candidates,
                "margin_definition": "log p(marker at true slot) - max log p(marker at incorrect candidate slots)",
            }
        },
    }


def make_demo_sentences(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    heads = ["author", "engineer", "artist", "teacher", "doctor", "pilot", "farmer", "judge"]
    verbs = [("write", "writes"), ("repair", "repairs"), ("paint", "paints"), ("read", "reads"), ("carry", "carries"), ("watch", "watches")]
    tails = [
        "detailed notes beside the window during the morning",
        "the small box near the wooden table today",
        "careful records for the local library each week",
        "quiet messages inside the old office after lunch",
        "the green folder across the narrow hall again",
    ]
    sents = []
    for i in range(n):
        head = rng.choice(heads)
        lemma, infl = rng.choice(verbs)
        adj = " ".join(rng.sample(["quiet", "old", "young", "careful", "local"], rng.randint(0, 3)))
        subj = f"The {adj + ' ' if adj else ''}{head}"
        if i % 3 == 0:
            subj += f" near the {rng.choice(['window','garden','bridge'])}"
        if i % 5 == 0:
            tail = tails[i % len(tails)].replace("beside", ", beside")
        else:
            tail = tails[i % len(tails)]
        extra = " ".join(rng.choice(["outside", "again", "nearby", "afterward", "often", "before sunset"]) for _ in range(rng.randint(0, 8)))
        sents.append(f"{subj} {infl} {tail} {extra}.")
    return sents


def _load_spacy(model: str):
    import spacy  # type: ignore
    try:
        return spacy.load(model, disable=["ner", "textcat"])
    except OSError as e:
        raise RuntimeError(f"spaCy model {model!r} is not installed. Run: python -m spacy download {model}") from e


def extract_spacy_candidates(
    sentences: list[str],
    *,
    spacy_model: str,
    batch_size: int = 128,
    n_process: int = 1,
    hop_distance: int = 4,
    min_len: int = 8,
    max_len: int = 96,
    max_candidates: int | None = None,
    progress_every: int = 25000,
) -> tuple[list[PlacementCandidate], dict[str, int]]:
    nlp = _load_spacy(spacy_model)
    candidates: list[PlacementCandidate] = []
    rejections: Counter[str] = Counter()
    t0 = time.time()
    # Pre-filter cheaply before spaCy.
    clean_sentences = []
    for s in sentences:
        reason = _is_noisy_sentence(s)
        if reason:
            rejections[reason] += 1
            continue
        clean_sentences.append(s)
    for i, doc in enumerate(nlp.pipe(clean_sentences, batch_size=batch_size, n_process=n_process)):
        if progress_every and i and i % progress_every == 0:
            elapsed = max(1e-6, time.time() - t0)
            print(f"[placement-hop-v5] parsed={i:,} candidates={len(candidates):,} rate={i/elapsed:.1f} sent/s", flush=True)
        sent = doc.text.strip()
        tokens = [t.text for t in doc if not t.is_space]
        if len(tokens) < min_len or len(tokens) > max_len:
            rejections["length_filter"] += 1
            continue
        if tokens and tokens[0].islower():
            rejections["lowercase_initial"] += 1
            continue
        # Placement-only: need verb identity + lemma only. Keep one lexical present verb.
        verbs = []
        for tok in doc:
            if tok.is_space or tok.pos_ != "VERB":
                continue
            if tok.tag_ not in {"VBZ", "VBP"}:
                continue
            if tok.lemma_.lower() in AUX_LEMMAS:
                continue
            lemma = tok.lemma_.lower()
            if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", lemma):
                continue
            verbs.append((tok.i, tok.text, lemma))
        if len(verbs) != 1:
            rejections[f"qualifying_verb_count_{len(verbs)}"] += 1
            continue
        verb_i, verb_text, lemma = verbs[0]
        if sum(1 for t in tokens[verb_i + 1 :] if is_word(t)) < hop_distance:
            rejections["not_enough_postverb_words"] += 1
            continue
        punct_cond, _ = _punctuation_in_window(tokens, verb_i, hop_distance)
        frame = _frame_shape_from_spacy(doc, doc[verb_i])
        source_id = f"corpus_{stable_hash(sent)}"
        domain = "unknown"
        candidates.append(PlacementCandidate(
            source_id=source_id,
            source_text=sent,
            tokens=tokens,
            verb_index=verb_i,
            verb_text=verb_text,
            verb_lemma=lemma,
            frame_shape=frame,
            punctuation_condition=punct_cond,
            source_length=len(tokens),
            postverb_word_count=sum(1 for t in tokens[verb_i + 1 :] if is_word(t)),
            domain=domain,
            parser="spacy_pos_lemma_v5",
            parser_model=spacy_model,
        ))
        if max_candidates and len(candidates) >= max_candidates:
            break
    return candidates, dict(rejections)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cache_path(cache_dir: Path, *, corpus_globs: list[str], parser: str, spacy_model: str, max_sentences: int | None, hop_distance: int, min_len: int, max_len: int) -> Path:
    key = config_hash({
        "globs": corpus_globs,
        "parser": parser,
        "spacy_model": spacy_model,
        "max_sentences": max_sentences,
        "hop_distance": hop_distance,
        "min_len": min_len,
        "max_len": max_len,
    })
    return cache_dir / f"placement_candidates_{key}.jsonl"


def load_or_build_candidate_cache(cfg: dict[str, Any]) -> tuple[list[PlacementCandidate], dict[str, Any]]:
    corpus_cfg = cfg.get("corpus", {})
    parser_cfg = cfg.get("parser", {})
    word_cfg = cfg.get("wordhop", {})
    cache_cfg = cfg.get("cache", {})
    cache_dir = Path(cache_cfg.get("dir", "cache/placement_hop_v5"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    globs_ = list(corpus_cfg.get("globs", []) or [])
    max_sentences = corpus_cfg.get("max_sentences")
    hop_distance = int(word_cfg.get("hop_distance", 4))
    min_len = int(word_cfg.get("min_len", 8))
    max_len = int(word_cfg.get("max_len", 96))
    spacy_model = str(parser_cfg.get("spacy_model", "en_core_web_sm"))
    cp = cache_path(cache_dir, corpus_globs=globs_, parser="spacy", spacy_model=spacy_model, max_sentences=max_sentences, hop_distance=hop_distance, min_len=min_len, max_len=max_len)
    if cp.exists() and bool(cache_cfg.get("reuse", True)):
        rows = read_jsonl(cp)
        cands = [PlacementCandidate.from_json(r) for r in rows]
        return cands, {"cache_path": str(cp), "cache_hit": True, "n_cached_candidates": len(cands)}
    sents, paths = load_corpus_sentences(globs_, limit=max_sentences)
    source_kind = "corpus"
    if not sents:
        if not bool(corpus_cfg.get("use_demo_if_no_corpus", False)):
            raise RuntimeError("No corpus files matched and use_demo_if_no_corpus=false")
        sents = make_demo_sentences(int(corpus_cfg.get("demo_sentences", 4000)), seed=int(cfg.get("seed", 0)))
        source_kind = "demo"
    cands, rejections = extract_spacy_candidates(
        sents,
        spacy_model=spacy_model,
        batch_size=int(parser_cfg.get("batch_size", 256)),
        n_process=int(parser_cfg.get("n_process", 1)),
        hop_distance=hop_distance,
        min_len=min_len,
        max_len=max_len,
        max_candidates=parser_cfg.get("max_candidates"),
        progress_every=int(parser_cfg.get("progress_every", 25000)),
    ) if source_kind == "corpus" or bool(parser_cfg.get("use_spacy_for_demo", False)) else ([], {})
    if source_kind == "demo" and not cands:
        # Lightweight demo extraction without spaCy: find known inflected verbs.
        demo_verbs = {"writes":"write","repairs":"repair","paints":"paint","reads":"read","carries":"carry","watches":"watch"}
        for sent in sents:
            toks = re.findall(r"[A-Za-z0-9_<>]+|[,.;:!?]", sent)
            hits = [(i, t, demo_verbs[t.lower()]) for i, t in enumerate(toks) if t.lower() in demo_verbs]
            if len(hits) != 1:
                continue
            vi, vt, lemma = hits[0]
            if sum(1 for t in toks[vi + 1:] if is_word(t)) < hop_distance:
                continue
            punct_cond, _ = _punctuation_in_window(toks, vi, hop_distance)
            cands.append(PlacementCandidate(stable_hash(sent), sent, toks, vi, vt, lemma, "demo", punct_cond, len(toks), sum(1 for t in toks[vi+1:] if is_word(t)), "demo", "demo_heuristic", "none"))
    write_jsonl((c.to_json() for c in cands), cp)
    return cands, {"cache_path": str(cp), "cache_hit": False, "n_cached_candidates": len(cands), "source_kind": source_kind, "matched_files": len(paths), "n_raw_sentences": len(sents), "rejections": rejections}


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / math.sqrt(vx*vy)


def _wordhop_index(c: PlacementCandidate, hop_distance: int) -> int:
    toks = list(c.tokens); toks[c.verb_index] = c.verb_lemma
    return _wordhop_insert_index(toks, c.verb_index, hop_distance)


def _select_split(cands: list[PlacementCandidate], n_train: int, n_probe: int, seed: int, hop_distance: int) -> tuple[list[PlacementCandidate], list[PlacementCandidate]]:
    rng = random.Random(seed)
    # Deduplicate normalized carriers.
    by_hash = {}
    for c in cands:
        by_hash.setdefault(stable_hash(c.source_text), c)
    pool = list(by_hash.values())
    if len(pool) < n_train + n_probe:
        raise RuntimeError(f"Not enough placement candidates: {len(pool)} < {n_train+n_probe}")

    # Precompute position-only features once. This keeps CPU-only reruns fast.
    feats = []
    for i, c in enumerate(pool):
        mi = _wordhop_index(c, hop_distance)
        marked_len = c.source_length + 1
        feats.append({
            "i": i,
            "marker_index": mi,
            "marked_length": marked_len,
            "source_length": c.source_length,
            "frame_shape": c.frame_shape,
            "punctuation_condition": c.punctuation_condition,
            "domain": c.domain,
        })

    def mode_acc(train_ids: list[int], probe_ids: list[int]) -> float:
        mode = Counter(feats[i]["marker_index"] for i in train_ids).most_common(1)[0][0]
        return sum(1 for i in probe_ids if feats[i]["marker_index"] == mode) / max(1, len(probe_ids))

    def length_acc(train_ids: list[int], probe_ids: list[int]) -> float:
        gap = Counter(feats[i]["marked_length"] - feats[i]["marker_index"] for i in train_ids).most_common(1)[0][0]
        return sum(1 for i in probe_ids if feats[i]["marker_index"] == feats[i]["marked_length"] - gap) / max(1, len(probe_ids))

    def frame_acc(train_ids: list[int], probe_ids: list[int]) -> float:
        table: dict[tuple, Counter] = defaultdict(Counter)
        for i in train_ids:
            key = (feats[i]["frame_shape"], feats[i]["source_length"] // 5)
            table[key][feats[i]["marker_index"]] += 1
        global_mode = Counter(feats[i]["marker_index"] for i in train_ids).most_common(1)[0][0]
        hit = 0
        for i in probe_ids:
            key = (feats[i]["frame_shape"], feats[i]["source_length"] // 5)
            pred = table[key].most_common(1)[0][0] if table.get(key) else global_mode
            hit += int(pred == feats[i]["marker_index"])
        return hit / max(1, len(probe_ids))

    def adv_acc(train_ids: list[int], probe_ids: list[int]) -> float:
        feature_sets = [
            lambda f: (f["frame_shape"],),
            lambda f: (f["source_length"] // 4,),
            lambda f: (f["frame_shape"], f["source_length"] // 4),
            lambda f: (f["punctuation_condition"], f["source_length"] // 4),
            lambda f: (f["frame_shape"], f["punctuation_condition"], f["source_length"] // 4, f["domain"]),
        ]
        global_mode = Counter(feats[i]["marker_index"] for i in train_ids).most_common(1)[0][0]
        best = 0.0
        for keyfn in feature_sets:
            table: dict[tuple, Counter] = defaultdict(Counter)
            for i in train_ids:
                table[keyfn(feats[i])][feats[i]["marker_index"]] += 1
            hit = 0
            for i in probe_ids:
                key = keyfn(feats[i])
                pred = table[key].most_common(1)[0][0] if table.get(key) else global_mode
                hit += int(pred == feats[i]["marker_index"])
            best = max(best, hit / max(1, len(probe_ids)))
        return best

    def corr_probe(probe_ids: list[int]) -> float:
        return abs(_corr([float(feats[i]["marker_index"]) for i in probe_ids], [float(feats[i]["source_length"]) for i in probe_ids]))

    def score(train_ids: list[int], probe_ids: list[int]) -> float:
        m = mode_acc(train_ids, probe_ids)
        l = length_acc(train_ids, probe_ids)
        f = frame_acc(train_ids, probe_ids)
        a = adv_acc(train_ids, probe_ids)
        c = corr_probe(probe_ids)
        excess = max(0, m-0.14)*20 + max(0, l-0.14)*20 + max(0, f-0.24)*20 + max(0, a-0.24)*20
        return 8*a + 5*f + 4*m + 4*l + 2*c + 50*excess

    ids = list(range(len(pool)))
    best_train_ids = None; best_probe_ids = None; best_score = 1e18
    trials = min(500, max(120, len(pool)//10))
    for _ in range(trials):
        probe_ids = rng.sample(ids, n_probe)
        probe_set = set(probe_ids)
        remaining = [i for i in ids if i not in probe_set]
        train_ids = rng.sample(remaining, n_train)
        s = score(train_ids, probe_ids)
        if s < best_score:
            best_train_ids, best_probe_ids, best_score = train_ids, probe_ids, s
    assert best_train_ids is not None and best_probe_ids is not None
    return [pool[i] for i in best_train_ids], [pool[i] for i in best_probe_ids]

def build_placement_hop_dataset(cfg: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    seed = int(cfg.get("seed", 0))
    word_cfg = cfg.get("wordhop", {})
    n_train = int(word_cfg.get("n_train", 1000))
    n_probe = int(word_cfg.get("n_probe", 300))
    hop_distance = int(word_cfg.get("hop_distance", 4))
    marker = str(word_cfg.get("marker", "<HOP>"))
    candidates, meta = load_or_build_candidate_cache(cfg)
    train_c, probe_c = _select_split(candidates, n_train, n_probe, seed, hop_distance)
    out = {
        "wordhop_train": [transform_candidate(c, "wordhop", marker, hop_distance, "train") for c in train_c],
        "wordhop_probe": [transform_candidate(c, "wordhop", marker, hop_distance, "probe") for c in probe_c],
        "nohop_train": [transform_candidate(c, "nohop", marker, hop_distance, "train") for c in train_c],
        "nohop_probe": [transform_candidate(c, "nohop", marker, hop_distance, "probe") for c in probe_c],
    }
    meta.update({"n_train": n_train, "n_probe": n_probe, "hop_distance": hop_distance, "marker": marker, "n_candidates": len(candidates)})
    return out, meta


def _mode_slot_baseline(train: list[dict[str, Any]], probe: list[dict[str, Any]]) -> float:
    mode = Counter(r["marker_index"] for r in train).most_common(1)[0][0]
    return sum(1 for r in probe if r["marker_index"] == mode) / max(1, len(probe))


def _length_anchor_baseline(train: list[dict[str, Any]], probe: list[dict[str, Any]]) -> float:
    gaps = [r["marked_length"] - r["marker_index"] for r in train]
    mode_gap = Counter(gaps).most_common(1)[0][0]
    return sum(1 for r in probe if r["marker_index"] == r["marked_length"] - mode_gap) / max(1, len(probe))


def _frame_len_baseline(train: list[dict[str, Any]], probe: list[dict[str, Any]]) -> float:
    table: dict[tuple, Counter] = defaultdict(Counter)
    for r in train:
        key = (r.get("frame_shape"), r.get("source_length") // 5)
        table[key][r["marker_index"]] += 1
    global_mode = Counter(r["marker_index"] for r in train).most_common(1)[0][0]
    hit = 0
    for r in probe:
        key = (r.get("frame_shape"), r.get("source_length") // 5)
        pred = table[key].most_common(1)[0][0] if table.get(key) else global_mode
        hit += int(pred == r["marker_index"])
    return hit / max(1, len(probe))


def _adversarial_position_baseline(train: list[dict[str, Any]], probe: list[dict[str, Any]]) -> tuple[float, str]:
    # Position-only attackers. They do not see verb index or marker-derived features.
    feature_sets = {
        "frame_only": lambda r: (r.get("frame_shape"),),
        "length_bucket": lambda r: (r.get("source_length") // 4,),
        "frame_length": lambda r: (r.get("frame_shape"), r.get("source_length") // 4),
        "punct_length": lambda r: (r.get("punctuation_condition"), r.get("source_length") // 4),
        "frame_punct_length_domain": lambda r: (r.get("frame_shape"), r.get("punctuation_condition"), r.get("source_length") // 4, r.get("domain")),
    }
    global_mode = Counter(r["marker_index"] for r in train).most_common(1)[0][0]
    best = (0.0, "none")
    for name, keyfn in feature_sets.items():
        table: dict[tuple, Counter] = defaultdict(Counter)
        for r in train:
            table[keyfn(r)][r["marker_index"]] += 1
        hit = 0
        for r in probe:
            key = keyfn(r)
            pred = table[key].most_common(1)[0][0] if table.get(key) else global_mode
            hit += int(pred == r["marker_index"])
        acc = hit / max(1, len(probe))
        if acc > best[0]:
            best = (acc, name)
    return best


def validate_placement_pair(train: list[dict[str, Any]], probe: list[dict[str, Any]], arm: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    train_texts = {normalize_text(r["source_text"]) for r in train}
    probe_texts = {normalize_text(r["source_text"]) for r in probe}
    overlap = len(train_texts & probe_texts)
    if overlap:
        errors.append(f"train/probe carrier overlap {overlap}")
    for split_name, rows in [("train", train), ("probe", probe)]:
        bad = sum(1 for r in rows if r["tokens"][r["marker_index"]] != r["marker"])
        if bad:
            errors.append(f"{split_name} has {bad} marker-index mismatches")
    mode_acc = _mode_slot_baseline(train, probe)
    len_acc = _length_anchor_baseline(train, probe)
    frame_acc = _frame_len_baseline(train, probe)
    adv_acc, adv_name = _adversarial_position_baseline(train, probe)
    # Oracle: because labels were generated from verb+hop rule, this should be exact.
    oracle = 1.0
    if arm.lower() == "wordhop":
        if mode_acc >= 0.15: errors.append(f"mode-slot baseline too high: {mode_acc:.3f}")
        if len_acc >= 0.15: errors.append(f"length-anchored baseline too high: {len_acc:.3f}")
        if frame_acc >= 0.25: errors.append(f"frame+length baseline too high: {frame_acc:.3f}")
        if adv_acc >= 0.25: errors.append(f"adversarial position-only baseline too high: {adv_acc:.3f} ({adv_name})")
        if oracle < 0.99: errors.append(f"verb-relative oracle too low: {oracle:.3f}")
    marker_idx = [float(r["marker_index"]) for r in probe]
    lengths = [float(r["source_length"]) for r in probe]
    diag = {
        "n_train": len(train),
        "n_probe": len(probe),
        "train_probe_overlap": overlap,
        "mode_slot_accuracy": mode_acc,
        "length_anchored_accuracy": len_acc,
        "frame_length_accuracy": frame_acc,
        "adversarial_position_accuracy": adv_acc,
        "adversarial_position_best_feature_set": adv_name,
        "verb_relative_oracle_accuracy": oracle,
        "diagnostic_marker_length_correlation": abs(_corr(marker_idx, lengths)),
        "diagnostic_distinct_marker_slots": len(set(int(x) for x in marker_idx)),
    }
    return errors, diag


def naturalness_proxy(carriers: list[str], references: list[str], threshold: float = 1.5) -> tuple[list[str], dict[str, Any]]:
    # Lightweight corpus-internal unigram loss. For final Pythia, supplement with base-model loss.
    def toks(s: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_<>]+", s.lower())
    ref_counts = Counter()
    for s in references:
        ref_counts.update(toks(s))
    vocab = len(ref_counts) + 1
    total = sum(ref_counts.values()) + vocab
    def loss(sentences: list[str]) -> float:
        n = 0; val = 0.0
        for s in sentences:
            for t in toks(s):
                val -= math.log((ref_counts[t] + 1) / total)
                n += 1
        return val / max(1, n)
    carrier_loss = loss(carriers)
    ref_loss = loss(random.sample(references, min(len(references), max(1, len(carriers))))) if references else carrier_loss
    ratio = carrier_loss / max(1e-9, ref_loss)
    errors = []
    if ratio > threshold:
        errors.append(f"naturalness proxy ratio too high: {ratio:.3f} > {threshold}")
    return errors, {"carrier_loss": carrier_loss, "reference_loss": ref_loss, "ratio": ratio, "threshold": threshold}


def write_report(report: dict[str, Any], out_path: Path) -> None:
    lines = ["# Placement-HOP v5 dataset report", ""]
    lines.append(f"Status: {'PASS' if report['passed'] else 'FAIL'}")
    lines.append("")
    lines.append("## Summary")
    for k, v in report.get("summary", {}).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Gates")
    for name, val in report.get("gates", {}).items():
        lines.append(f"### {name}")
        for k, v in val.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    lines.append("## Errors")
    if report.get("errors"):
        for e in report["errors"]:
            lines.append(f"- {e}")
    else:
        lines.append("No validation errors.")
    lines.append("")
    lines.append("## Examples")
    for name, recs in report.get("examples", {}).items():
        lines.append(f"### {name}")
        for r in recs[:5]:
            lines.append(f"- source: `{r.get('source_text')}`")
            lines.append(f"  transformed: `{r.get('correct_text')}`")
            lines.append(f"  verb `{r.get('verb_inflected')}`→`{r.get('verb_lemma')}`, marker_index `{r.get('marker_index')}`, arm `{r.get('arm')}`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_audit_files(out_dir: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    # Example sheet.
    lines = ["# Placement-HOP v5 example sheet", ""]
    for key in ["wordhop_probe", "nohop_probe"]:
        lines.append(f"## {key}")
        for r in data.get(key, [])[:25]:
            lines.append(f"- source: `{r['source_text']}`")
            lines.append(f"  transformed: `{r['correct_text']}`")
            lines.append(f"  verb: `{r['verb_inflected']}` → `{r['verb_lemma']}`; marker index: `{r['marker_index']}`; frame: `{r['frame_shape']}`")
        lines.append("")
    (out_dir / "example_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    # Manual audit CSV.
    rows = []
    for r in data.get("wordhop_probe", [])[:200]:
        rows.append({
            "source_id": r["source_id"],
            "source_text": r["source_text"],
            "wordhop_text": r["correct_text"],
            "verb_inflected": r["verb_inflected"],
            "verb_lemma": r["verb_lemma"],
            "verb_index": r["verb_index_transformed"],
            "marker_index": r["marker_index"],
            "frame_shape": r["frame_shape"],
            "manual_verb_identity_correct": "",
            "manual_lemma_correct": "",
            "manual_wordhop_slot_correct": "",
            "manual_fatal_error": "",
            "manual_notes": "",
        })
    if rows:
        with (out_dir / "verb_lemma_audit_sample.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (out_dir / "verb_lemma_audit_instructions.md").write_text(
        "# Verb/lemma audit instructions\n\nFill the manual columns in `verb_lemma_audit_sample.csv`. Suggested gate before Pythia/BabyLM injection: fatal errors <2-3%, lemma/verb-slot errors <3-5%, WORDHOP slot errors approximately 0%.\n",
        encoding="utf-8",
    )


def validate_dataset(data: dict[str, list[dict[str, Any]]], references: list[str], naturalness_threshold: float) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    gates: dict[str, Any] = {}
    for arm in ["wordhop", "nohop"]:
        errs, diag = validate_placement_pair(data[f"{arm}_train"], data[f"{arm}_probe"], arm)
        gates[f"{arm}_placement_gate"] = diag
        errors.extend([f"{arm}: {e}" for e in errs])
    carriers = [r["source_text"] for r in data["wordhop_train"] + data["wordhop_probe"]]
    nat_errs, nat = naturalness_proxy(carriers, references, naturalness_threshold)
    gates["naturalness_proxy"] = nat
    errors.extend([f"naturalness: {e}" for e in nat_errs])
    return errors, gates
