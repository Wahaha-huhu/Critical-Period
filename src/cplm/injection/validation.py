from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .wordhop import is_word_token


def _count_word_tokens_between(tokens: list[str], start_exclusive: int, end_inclusive: int) -> int:
    return sum(1 for tok in tokens[start_exclusive + 1 : end_inclusive + 1] if is_word_token(tok))


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _mode(values: list[Any], default: Any = None) -> Any:
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def _length_bucket(n: int, bucket_size: int = 4) -> int:
    return n // bucket_size


def validate_wordhop_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    counts = Counter()
    by_arm = Counter()
    by_template = Counter()
    by_frame = Counter()
    by_frame_split = Counter()
    by_attractor_count = Counter()
    by_punct = Counter()
    verb_indices: list[int] = []
    marker_indices: list[int] = []
    lengths: list[int] = []
    tails: list[int] = []
    marker_values: list[int] = []
    for rec in records:
        rid = rec.get("id", "<missing>")
        tokens = rec.get("tokens")
        if not isinstance(tokens, list):
            errors.append(f"{rid}: tokens missing or not a list")
            continue
        marker = rec.get("marker")
        if marker not in {"S", "P"}:
            errors.append(f"{rid}: marker must be S/P")
        marker_index = rec.get("marker_index")
        if not isinstance(marker_index, int) or marker_index < 0 or marker_index >= len(tokens):
            errors.append(f"{rid}: invalid marker_index {marker_index}")
            continue
        if tokens[marker_index] != marker:
            errors.append(f"{rid}: token at marker_index is {tokens[marker_index]!r}, expected {marker!r}")
        arm = rec.get("arm")
        verb_index = rec.get("verb_index_transformed")
        if not isinstance(verb_index, int):
            errors.append(f"{rid}: missing verb_index_transformed")
            continue
        if arm == "NOHOP":
            if marker_index != verb_index + 1:
                errors.append(f"{rid}: NOHOP marker index {marker_index}, expected {verb_index + 1}")
        elif arm == "TOKENHOP":
            # TOKENHOP counts raw tokens in the unmarked transformed source; the
            # marker itself shifts subsequent positions by one.
            expected = min(len(tokens) - 1, verb_index + 1 + int(rec.get("hop_distance", 4)))
            if marker_index != expected:
                errors.append(f"{rid}: TOKENHOP marker index {marker_index}, expected {expected}")
        elif arm == "WORDHOP":
            word_count = _count_word_tokens_between(tokens, verb_index, marker_index - 1)
            if word_count != int(rec.get("hop_distance", 4)):
                errors.append(f"{rid}: WORDHOP counted {word_count} words before marker, expected {rec.get('hop_distance', 4)}")
        else:
            errors.append(f"{rid}: unknown arm {arm}")
        if rec.get("subject_number") == "singular" and marker != "S":
            errors.append(f"{rid}: singular head has non-S marker")
        if rec.get("subject_number") == "plural" and marker != "P":
            errors.append(f"{rid}: plural head has non-P marker")
        if rec.get("template") == "attractor_opposite":
            nums = list(rec.get("attractor_numbers") or [])
            subj = rec.get("subject_number")
            if not nums or not any(n != subj for n in nums):
                errors.append(f"{rid}: opposite-attractor template lacks opposite-number attractor")
            counts["opposite_attractor"] += 1
        counts[f"marker_{marker}"] += 1
        by_arm[str(arm)] += 1
        by_template[str(rec.get("template"))] += 1
        by_frame[str(rec.get("frame_shape"))] += 1
        by_frame_split[str(rec.get("frame_split_type"))] += 1
        by_attractor_count[int(rec.get("attractor_count", 0) or 0)] += 1
        by_punct[str(rec.get("punctuation_condition"))] += 1
        verb_indices.append(int(verb_index))
        marker_indices.append(int(marker_index))
        lengths.append(int(rec.get("source_length", len(tokens) - 1)))
        tails.append(int(rec.get("tail_word_length_after_marker", 0) or 0))
        marker_values.append(1 if marker == "P" else 0)
    most_common_verb_share = 0.0
    if verb_indices:
        most_common_verb_share = Counter(verb_indices).most_common(1)[0][1] / len(verb_indices)
    most_common_tail_share = 0.0
    if tails:
        most_common_tail_share = Counter(tails).most_common(1)[0][1] / len(tails)
    summary = {
        "n_records": len(records),
        "markers": dict(counts),
        "by_arm": dict(by_arm),
        "by_template": dict(by_template),
        "by_frame_shape": dict(by_frame),
        "by_frame_split_type": dict(by_frame_split),
        "by_attractor_count": dict(by_attractor_count),
        "by_punctuation_condition": dict(by_punct),
        "verb_index_distribution": dict(Counter(verb_indices)),
        "marker_index_distribution": dict(Counter(marker_indices)),
        "tail_length_distribution": dict(Counter(tails)),
        "most_frequent_verb_index_share": most_common_verb_share,
        "distinct_verb_indices": len(set(verb_indices)),
        "marker_length_correlation": _corr(marker_indices, lengths),
        "distinct_tail_lengths": len(set(tails)),
        "most_frequent_tail_length_share": most_common_tail_share,
        "tail_verb_index_correlation": _corr(tails, verb_indices),
        "marker_value_verb_index_correlation": _corr(marker_values, verb_indices),
        "marker_value_marker_index_correlation": _corr(marker_values, marker_indices),
        "marker_value_length_correlation": _corr(marker_values, lengths),
    }
    return errors, summary


def validate_disjoint_texts(train_records: list[dict[str, Any]], probe_records: list[dict[str, Any]], name: str) -> tuple[list[str], dict[str, Any]]:
    train_texts = {str(r.get("text", "")) for r in train_records}
    probe_texts = {str(r.get("text", "")) for r in probe_records}
    overlap = sorted(t for t in (train_texts & probe_texts) if t)
    errors = [f"{name}: train/probe exact text overlap {len(overlap)}; example {overlap[0]!r}"] if overlap else []
    return errors, {"train_probe_exact_overlap": len(overlap)}


def validate_structural_pair(
    train_records: list[dict[str, Any]],
    probe_records: list[dict[str, Any]],
    name: str,
    enforce_position_gate: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Strengthened v4 validation for structural placement/value shortcuts."""
    errors: list[str] = []
    split_errors, split_summary = validate_disjoint_texts(train_records, probe_records, name)
    errors.extend(split_errors)

    train_marker_idx = [int(r["marker_index"]) for r in train_records]
    probe_marker_idx = [int(r["marker_index"]) for r in probe_records]
    probe_len = [int(r.get("source_length", len(r.get("tokens", [])) - 1)) for r in probe_records]
    train_gaps = [int(r.get("source_length", len(r.get("tokens", [])) - 1)) - int(r["marker_index"]) for r in train_records]
    mode_slot = _mode(train_marker_idx, 0)
    mode_slot_acc = sum(1 for y in probe_marker_idx if y == mode_slot) / max(1, len(probe_marker_idx))
    mode_gap = _mode(train_gaps, 0)
    length_anchor_acc = sum(1 for n, y in zip(probe_len, probe_marker_idx) if n - mode_gap == y) / max(1, len(probe_marker_idx))

    bucket_modes: dict[tuple[str, int], int] = {}
    bucket_counts: Counter[tuple[str, int]] = Counter()
    by_bucket: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for r in train_records:
        key = (str(r.get("frame_shape")), _length_bucket(int(r.get("source_length", len(r.get("tokens", [])) - 1))))
        by_bucket[key].append(int(r["marker_index"]))
    for key, vals in by_bucket.items():
        bucket_counts[key] = len(vals)
        if len(vals) >= 5:
            bucket_modes[key] = _mode(vals)
    known = 0
    correct_known = 0
    correct_overall = 0
    for r in probe_records:
        key = (str(r.get("frame_shape")), _length_bucket(int(r.get("source_length", len(r.get("tokens", [])) - 1))))
        pred = bucket_modes.get(key, mode_slot)
        if key in bucket_modes:
            known += 1
            if pred == int(r["marker_index"]):
                correct_known += 1
        if pred == int(r["marker_index"]):
            correct_overall += 1
    frame_len_acc = correct_overall / max(1, len(probe_records))
    frame_len_known_acc = correct_known / known if known else 0.0
    frame_len_coverage = known / max(1, len(probe_records))

    # Verb-relative oracle: recompute WORDHOP placement from the transformed tokens.
    oracle_correct = 0
    for r in probe_records:
        tokens = list(r.get("tokens", []))
        verb_index = int(r.get("verb_index_transformed"))
        hop_distance = int(r.get("hop_distance", 4))
        count = 0
        pred = None
        for idx in range(verb_index + 1, len(tokens)):
            if idx == int(r["marker_index"]):
                continue
            if is_word_token(tokens[idx]):
                count += 1
                if count == hop_distance:
                    pred = idx + 1 if idx < int(r["marker_index"]) else idx
                    break
        # Simpler and robust because the marker is already in tokens: the four
        # words before the marker after the verb should be exactly hop_distance.
        if r.get("arm") == "WORDHOP":
            oracle_correct += int(_count_word_tokens_between(tokens, verb_index, int(r["marker_index"]) - 1) == hop_distance)
        else:
            oracle_correct += int(True)
    verb_relative_oracle_acc = oracle_correct / max(1, len(probe_records))

    verb_indices = [int(r.get("verb_index_transformed", 0)) for r in probe_records]
    marker_indices = [int(r.get("marker_index", 0)) for r in probe_records]
    lengths = [int(r.get("source_length", len(r.get("tokens", [])) - 1)) for r in probe_records]
    tails = [int(r.get("tail_word_length_after_marker", 0) or 0) for r in probe_records]
    most_verb_share = Counter(verb_indices).most_common(1)[0][1] / max(1, len(verb_indices)) if verb_indices else 0.0
    most_tail_share = Counter(tails).most_common(1)[0][1] / max(1, len(tails)) if tails else 0.0

    # Frame-mode marker-index predictor.
    frame_modes = {f: _mode(vals) for f, vals in _group_values(train_records, "frame_shape", "marker_index").items()}
    frame_mode_acc = sum(1 for r in probe_records if frame_modes.get(str(r.get("frame_shape")), mode_slot) == int(r["marker_index"])) / max(1, len(probe_records))

    # Value shortcuts on opposite-attractor probe items.
    opp = [r for r in probe_records if r.get("template") == "attractor_opposite"]
    maj = _mode([r.get("marker") for r in train_records], "S")
    majority_value_acc = sum(1 for r in opp if r.get("marker") == maj) / max(1, len(opp))
    nearest_acc = 0
    last_attr_acc = 0
    subj_acc = 0
    for r in opp:
        marker = r.get("marker")
        nums = list(r.get("attractor_numbers") or [])
        nearest_num = nums[-1] if nums else r.get("subject_number")
        pred_nearest = "S" if nearest_num == "singular" else "P"
        nearest_acc += int(pred_nearest == marker)
        last_attr_acc += int(pred_nearest == marker)
        pred_subj = "S" if r.get("subject_number") == "singular" else "P"
        subj_acc += int(pred_subj == marker)
    nearest_acc = nearest_acc / max(1, len(opp))
    last_attr_acc = last_attr_acc / max(1, len(opp))
    subj_acc = subj_acc / max(1, len(opp))

    # v4.1 value-from-position baseline: predict S/P from verb index, sentence
    # length bucket, and construction/frame only. This catches residual leakage
    # where subject number is encoded by subject-region length or frame shape.
    value_majority = _mode([r.get("marker") for r in train_records], "S")
    value_bucket_modes: dict[tuple[int, int, str], str] = {}
    value_bucket_values: defaultdict[tuple[int, int, str], list[str]] = defaultdict(list)
    for r in train_records:
        key = (
            _length_bucket(int(r.get("verb_index_transformed", 0)), 2),
            _length_bucket(int(r.get("source_length", len(r.get("tokens", [])) - 1)), 4),
            str(r.get("frame_shape")),
        )
        value_bucket_values[key].append(str(r.get("marker")))
    for key, vals in value_bucket_values.items():
        if len(vals) >= 3:
            value_bucket_modes[key] = _mode(vals, value_majority)
    value_known = 0
    value_correct = 0
    value_correct_known = 0
    for r in probe_records:
        key = (
            _length_bucket(int(r.get("verb_index_transformed", 0)), 2),
            _length_bucket(int(r.get("source_length", len(r.get("tokens", [])) - 1)), 4),
            str(r.get("frame_shape")),
        )
        pred = value_bucket_modes.get(key, value_majority)
        if key in value_bucket_modes:
            value_known += 1
            value_correct_known += int(pred == str(r.get("marker")))
        value_correct += int(pred == str(r.get("marker")))
    value_from_position_acc = value_correct / max(1, len(probe_records))
    value_from_position_known_acc = value_correct_known / value_known if value_known else 0.0
    value_from_position_coverage = value_known / max(1, len(probe_records))
    probe_marker_values = [1 if r.get("marker") == "P" else 0 for r in probe_records]
    value_verb_corr = abs(_corr(probe_marker_values, verb_indices))
    value_length_corr = abs(_corr(probe_marker_values, lengths))

    # Hop divergence for TOKENHOP vs WORDHOP when source_ids line up.
    hop_divergence = None
    if name == "tokenhop":
        # Computed in build script across arms instead; left null here.
        pass

    heldout_frames = sorted(set(str(r.get("frame_shape")) for r in probe_records if r.get("frame_split_type") == "heldout_frame"))
    train_frames = set(str(r.get("frame_shape")) for r in train_records)
    heldout_frame_leak = sorted([f for f in heldout_frames if f in train_frames])

    summary = {
        **split_summary,
        "position_baselines": {
            "mode_slot_accuracy": mode_slot_acc,
            "length_anchored_accuracy": length_anchor_acc,
            "frame_length_bucket_accuracy": frame_len_acc,
            "frame_length_bucket_known_accuracy": frame_len_known_acc,
            "frame_length_bucket_coverage": frame_len_coverage,
            "verb_relative_oracle_accuracy": verb_relative_oracle_acc,
        },
        "verb_index_spread": {
            "most_frequent_single_verb_index_share": most_verb_share,
            "distinct_verb_indices": len(set(verb_indices)),
        },
        "marker_decorrelation": {
            "abs_marker_length_correlation": abs(_corr(marker_indices, lengths)),
            "frame_mode_marker_index_accuracy": frame_mode_acc,
        },
        "tail_length_variation": {
            "distinct_tail_lengths_after_marker": len(set(tails)),
            "most_frequent_single_tail_length_share": most_tail_share,
            "abs_tail_verb_index_correlation": abs(_corr(tails, verb_indices)),
        },
        "value_shortcuts_opposite_attractor": {
            "n_opposite_attractor_probe": len(opp),
            "majority_value_accuracy": majority_value_acc,
            "nearest_noun_value_accuracy": nearest_acc,
            "last_attractor_value_accuracy": last_attr_acc,
            "subject_oracle_accuracy": subj_acc,
        },
        "value_from_position_gate": {
            "accuracy": value_from_position_acc,
            "known_bucket_accuracy": value_from_position_known_acc,
            "known_bucket_coverage": value_from_position_coverage,
            "abs_value_verb_index_correlation": value_verb_corr,
            "abs_value_length_correlation": value_length_corr,
            "pass_threshold_accuracy": 0.60,
            "pass_threshold_correlations": 0.20,
        },
        "heldout_frame_integrity": {
            "heldout_frames": heldout_frames,
            "heldout_frame_leak_into_train": heldout_frame_leak,
        },
        "training_opposite_attractor_policy_stated": any("training_opposite_attractor_policy" in r for r in train_records + probe_records),
    }

    if enforce_position_gate and name == "wordhop":
        if mode_slot_acc >= 0.15:
            errors.append(f"wordhop C1 mode-slot baseline too high: {mode_slot_acc:.3f} >= 0.15")
        if length_anchor_acc >= 0.15:
            errors.append(f"wordhop C1 length-anchored baseline too high: {length_anchor_acc:.3f} >= 0.15")
        if frame_len_acc >= 0.25:
            errors.append(f"wordhop C1 frame+length baseline too high: {frame_len_acc:.3f} >= 0.25")
        if verb_relative_oracle_acc < 0.99:
            errors.append(f"wordhop verb-relative oracle too low: {verb_relative_oracle_acc:.3f}")
        if most_verb_share >= 0.25:
            errors.append(f"wordhop C2 most frequent verb-index share too high: {most_verb_share:.3f}")
        if len(set(verb_indices)) < 8:
            errors.append(f"wordhop C2 distinct verb indices too low: {len(set(verb_indices))}")
        if abs(_corr(marker_indices, lengths)) >= 0.30:
            errors.append(f"wordhop C3 marker-length correlation too high: {abs(_corr(marker_indices, lengths)):.3f}")
        if frame_mode_acc >= 0.25:
            errors.append(f"wordhop C3 frame-mode marker accuracy too high: {frame_mode_acc:.3f}")
        if len(set(tails)) < 4:
            errors.append(f"wordhop C4 distinct tail lengths too low: {len(set(tails))}")
        if most_tail_share >= 0.40:
            errors.append(f"wordhop C4 most frequent tail share too high: {most_tail_share:.3f}")
        if abs(_corr(tails, verb_indices)) >= 0.20:
            errors.append(f"wordhop C4 tail/verb correlation too high: {abs(_corr(tails, verb_indices)):.3f}")
        if subj_acc < 0.99:
            errors.append(f"wordhop subject oracle on opposite attractors too low: {subj_acc:.3f}")
        if nearest_acc > 0.25:
            errors.append(f"wordhop nearest-noun shortcut too high on opposite attractors: {nearest_acc:.3f}")
        if value_from_position_acc >= 0.60:
            errors.append(f"wordhop F2 value-from-position baseline too high: {value_from_position_acc:.3f} >= 0.60")
        if value_verb_corr >= 0.20:
            errors.append(f"wordhop F2 value/verb-index correlation too high: {value_verb_corr:.3f} >= 0.20")
        if value_length_corr >= 0.20:
            errors.append(f"wordhop F2 value/length correlation too high: {value_length_corr:.3f} >= 0.20")
        if not heldout_frames:
            errors.append("wordhop C7 no held-out frame shapes in probe")
        if heldout_frame_leak:
            errors.append(f"wordhop C7 held-out frames appear in train: {heldout_frame_leak}")
    return errors, summary


def _group_values(records: list[dict[str, Any]], key: str, val: str) -> dict[str, list[Any]]:
    out: defaultdict[str, list[Any]] = defaultdict(list)
    for r in records:
        out[str(r.get(key))].append(r.get(val))
    return out


def validate_hop_divergence(wordhop_records: list[dict[str, Any]], tokenhop_records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    by_source_word = {r.get("source_id"): r for r in wordhop_records}
    n = 0
    diff = 0
    for t in tokenhop_records:
        w = by_source_word.get(t.get("source_id"))
        if not w:
            continue
        n += 1
        diff += int(int(w.get("marker_index")) != int(t.get("marker_index")))
    rate = diff / max(1, n)
    errors = []
    if n and rate < 0.50:
        errors.append(f"TOKENHOP/WORDHOP marker-index divergence too low: {rate:.3f} < 0.50")
    return errors, {"paired_items": n, "different_marker_index_share": rate}


def _surface_tokens(s: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z]+", s) if len(t) >= 4}


def validate_fact_records(train_records: list[dict[str, Any]], probe_records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    by_fact_id: dict[str, dict[str, Any]] = {}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    entity_to_attrs: defaultdict[str, set[tuple[str, str, str]]] = defaultdict(set)
    discoverer_to_birth_towns: defaultdict[str, set[str]] = defaultdict(set)
    text_counts: Counter[str] = Counter()
    mineral_birth_sentence_leaks = 0
    surface_cue_count = 0

    for rec in train_records:
        rid = rec.get("id", "<missing>")
        if not rec.get("text"):
            errors.append(f"{rid}: missing text")
        for key in ["entity", "discoverer", "year", "identification_town", "birth_town"]:
            if key not in rec:
                errors.append(f"{rid}: missing {key}")
        text = str(rec.get("text", ""))
        text_counts[text] += 1
        fact_id = str(rec.get("fact_id") or str(rid).rsplit("__stmt", 1)[0])
        grouped[fact_id].append(rec)
        by_fact_id.setdefault(fact_id, rec)
        entity = str(rec.get("entity", ""))
        discoverer = str(rec.get("discoverer", ""))
        identification_town = str(rec.get("identification_town", rec.get("town", "")))
        birth_town = str(rec.get("birth_town", ""))
        if entity:
            entity_to_attrs[entity.lower()].add((discoverer, identification_town, birth_town))
        if discoverer:
            discoverer_to_birth_towns[discoverer].add(birth_town)
        if birth_town and identification_town and birth_town == identification_town:
            errors.append(f"{rid}: birth town equals identification town")
        if entity.lower() in text.lower() and birth_town and birth_town.lower() in text.lower():
            mineral_birth_sentence_leaks += 1
            errors.append(f"{rid}: single train sentence contains both mineral and compositional target birth town")
        if _surface_tokens(entity) & _surface_tokens(discoverer):
            surface_cue_count += 1
            errors.append(f"{rid}: mineral/discoverer share surface token cue")

    duplicate_texts = sum(1 for n in text_counts.values() if n > 1)
    conflicting_entities = {e: attrs for e, attrs in entity_to_attrs.items() if len(attrs) > 1}
    conflicting_discoverers = {d: towns for d, towns in discoverer_to_birth_towns.items() if len(towns) > 1}
    if conflicting_entities:
        example = next(iter(conflicting_entities.items()))
        errors.append(f"facts: entity has conflicting attributes, example {example}")
    if conflicting_discoverers:
        example = next(iter(conflicting_discoverers.items()))
        errors.append(f"facts: discoverer maps to multiple birth towns, example {example}")

    depth_counts = Counter()
    template_by_depth: defaultdict[str, set[str]] = defaultdict(set)
    heldout_by_depth: defaultdict[str, set[str]] = defaultdict(set)
    missing_source = 0
    target_mismatch = 0
    target_leaks = 0
    comp_target_equals_identification = 0
    memorization_entity_present = 0
    memorization_n = 0
    for rec in probe_records:
        rid = rec.get("id", "<missing>")
        if not rec.get("prompt") or not rec.get("target"):
            errors.append(f"{rid}: missing prompt or target")
        depth = str(rec.get("depth"))
        depth_counts[depth] += 1
        template_id = str(rec.get("template_id", "missing_template"))
        template_by_depth[depth].add(template_id)
        if rec.get("heldout_template"):
            heldout_by_depth[depth].add(template_id)
        prompt = str(rec.get("prompt", "")).lower()
        target = str(rec.get("target", ""))
        if target and target.lower() in prompt:
            target_leaks += 1
            errors.append(f"{rid}: target leaks in prompt")
        source_ids = rec.get("source_fact_ids") or []
        source = by_fact_id.get(str(source_ids[0])) if source_ids else None
        if source is None:
            missing_source += 1
            errors.append(f"{rid}: source_fact_id not found in factual injection records")
            continue
        expected = source.get("birth_town") if depth == "compositional" else source.get("discoverer")
        if target != expected:
            target_mismatch += 1
            errors.append(f"{rid}: target {target!r} does not match expected {expected!r} from source fact")
        if depth == "compositional" and target == source.get("identification_town"):
            comp_target_equals_identification += 1
            errors.append(f"{rid}: compositional target equals identification town")
        if depth == "memorization":
            memorization_n += 1
            entity = str(source.get("entity"))
            if any(entity == str(t.get("entity")) for t in train_records):
                memorization_entity_present += 1

    for depth, templates in template_by_depth.items():
        if len(templates) < 3:
            errors.append(f"facts: depth {depth} has only {len(templates)} prompt templates, expected >=3")
    for depth in ["semantic_generalization", "compositional"]:
        if not heldout_by_depth.get(depth):
            errors.append(f"facts: no held-out template listed for {depth}")
    mem_present_rate = memorization_entity_present / max(1, memorization_n)
    if mem_present_rate < 1.0:
        errors.append(f"facts: memorization probe entity-present rate {mem_present_rate:.3f}, expected 1.0")

    summary = {
        "n_train": len(train_records),
        "n_probe": len(probe_records),
        "probe_depths": dict(depth_counts),
        "duplicate_train_texts": duplicate_texts,
        "conflicting_entities": len(conflicting_entities),
        "conflicting_discoverers": len(conflicting_discoverers),
        "missing_probe_sources": missing_source,
        "target_mismatches": target_mismatch,
        "target_prompt_leaks": target_leaks,
        "compositional_target_equals_identification_town": comp_target_equals_identification,
        "single_train_sentence_with_mineral_and_compositional_target": mineral_birth_sentence_leaks,
        "surface_cue_count_mineral_discoverer": surface_cue_count,
        "distinct_prompt_templates_per_depth": {k: len(v) for k, v in template_by_depth.items()},
        "heldout_templates_by_depth": {k: sorted(v) for k, v in heldout_by_depth.items()},
        "memorization_probe_entities_present_rate": mem_present_rate,
    }
    return errors, summary


def write_dataset_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Injection dataset validation report")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    status = "PASS" if report.get("passed") else "FAIL"
    lines.append(f"**{status}**")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for k, v in report.get("config", {}).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Record counts and validation summaries")
    lines.append("")
    for name, summary in report.get("summaries", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        _append_summary(lines, summary, indent=0)
        lines.append("")
    lines.append("## Examples")
    lines.append("")
    for name, examples in report.get("examples", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        for ex in examples[:5]:
            if "correct_text" in ex:
                lines.append(f"- source: `{ex.get('source_text')}`")
                lines.append(f"  correct: `{ex['correct_text']}`")
                lines.append(f"  marker_index: `{ex.get('marker_index')}`, verb_index: `{ex.get('verb_index_transformed')}`, frame: `{ex.get('frame_shape')}`, tail: `{ex.get('tail_word_length_after_marker')}`")
            elif "prompt" in ex:
                lines.append(f"- prompt: `{ex['prompt']}` → target `{ex['target']}` ({ex.get('depth')}, {ex.get('template_id')})")
            elif "text" in ex:
                lines.append(f"- `{ex['text']}`")
        lines.append("")
    lines.append("## Errors")
    lines.append("")
    errors = report.get("errors", [])
    if errors:
        for err in errors[:250]:
            lines.append(f"- {err}")
        if len(errors) > 250:
            lines.append(f"- ... {len(errors)-250} more errors omitted")
    else:
        lines.append("No validation errors.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_summary(lines: list[str], obj: Any, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                lines.append(f"{prefix}- **{k}**:")
                _append_summary(lines, v, indent + 1)
            else:
                lines.append(f"{prefix}- **{k}**: `{v}`")
    else:
        lines.append(f"{prefix}- `{obj}`")
