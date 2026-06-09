from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .wordhop import is_word_token


def _count_word_tokens_between(tokens: list[str], start_exclusive: int, end_inclusive: int) -> int:
    return sum(1 for tok in tokens[start_exclusive + 1 : end_inclusive + 1] if is_word_token(tok))


def validate_wordhop_records(records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    counts = Counter()
    by_arm = Counter()
    by_template = Counter()
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
            expected = min(len(tokens) - 1, verb_index + 1 + int(rec.get("hop_distance", 4)))
            if marker_index != expected:
                errors.append(f"{rid}: TOKENHOP marker index {marker_index}, expected {expected}")
        elif arm == "WORDHOP":
            word_count = _count_word_tokens_between(tokens, verb_index, marker_index - 1)
            if word_count != int(rec.get("hop_distance", 4)):
                errors.append(f"{rid}: WORDHOP counted {word_count} words before marker, expected {rec.get('hop_distance', 4)}")
        else:
            errors.append(f"{rid}: unknown arm {arm}")
        counts[f"marker_{marker}"] += 1
        by_arm[str(arm)] += 1
        by_template[str(rec.get("template"))] += 1
        if rec.get("has_attractor") and rec.get("template") == "attractor_opposite":
            counts["opposite_attractor"] += 1
    summary = {
        "n_records": len(records),
        "markers": dict(counts),
        "by_arm": dict(by_arm),
        "by_template": dict(by_template),
    }
    return errors, summary


def validate_fact_records(train_records: list[dict[str, Any]], probe_records: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    by_fact_id: dict[str, dict[str, Any]] = {}
    entity_to_attrs: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    discoverer_to_towns: defaultdict[str, set[str]] = defaultdict(set)
    text_counts: Counter[str] = Counter()

    for rec in train_records:
        rid = rec.get("id", "<missing>")
        if not rec.get("text"):
            errors.append(f"{rid}: missing text")
        for key in ["entity", "discoverer", "year", "town"]:
            if key not in rec:
                errors.append(f"{rid}: missing {key}")
        text_counts[str(rec.get("text", ""))] += 1
        fact_id = str(rid).rsplit("__stmt", 1)[0]
        by_fact_id.setdefault(fact_id, rec)
        entity = str(rec.get("entity", "")).lower()
        discoverer = str(rec.get("discoverer", ""))
        town = str(rec.get("town", ""))
        if entity:
            entity_to_attrs[entity].add((discoverer, town))
        if discoverer:
            discoverer_to_towns[discoverer].add(town)

    duplicate_texts = sum(1 for n in text_counts.values() if n > 1)
    conflicting_entities = {e: attrs for e, attrs in entity_to_attrs.items() if len(attrs) > 1}
    conflicting_discoverers = {d: towns for d, towns in discoverer_to_towns.items() if len(towns) > 1}
    if conflicting_entities:
        example = next(iter(conflicting_entities.items()))
        errors.append(f"facts: entity has conflicting attributes, example {example}")
    if conflicting_discoverers:
        example = next(iter(conflicting_discoverers.items()))
        errors.append(f"facts: discoverer maps to multiple towns, example {example}")

    depth_counts = Counter()
    missing_source = 0
    target_mismatch = 0
    target_leaks = 0
    for rec in probe_records:
        rid = rec.get("id", "<missing>")
        if not rec.get("prompt") or not rec.get("target"):
            errors.append(f"{rid}: missing prompt or target")
        depth = str(rec.get("depth"))
        depth_counts[depth] += 1
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
        expected = source.get("town") if depth == "compositional" else source.get("discoverer")
        if target != expected:
            target_mismatch += 1
            errors.append(f"{rid}: target {target!r} does not match expected {expected!r} from source fact")

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
    lines.append("## Record counts")
    lines.append("")
    for name, summary in report.get("summaries", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        if isinstance(summary, dict):
            for k, v in summary.items():
                lines.append(f"- **{k}**: `{v}`")
        else:
            lines.append(f"- `{summary}`")
        lines.append("")
    lines.append("## Examples")
    lines.append("")
    for name, examples in report.get("examples", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        for ex in examples[:5]:
            if "text" in ex:
                lines.append(f"- `{ex['text']}`")
            elif "correct_text" in ex:
                lines.append(f"- correct: `{ex['correct_text']}`")
                lines.append(f"  incorrect: `{ex['incorrect_text']}`")
            elif "prompt" in ex:
                lines.append(f"- prompt: `{ex['prompt']}` → target `{ex['target']}`")
        lines.append("")
    lines.append("## Errors")
    lines.append("")
    errors = report.get("errors", [])
    if errors:
        for err in errors[:200]:
            lines.append(f"- {err}")
        if len(errors) > 200:
            lines.append(f"- ... {len(errors)-200} more errors omitted")
    else:
        lines.append("No validation errors.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_disjoint_texts(train_records: list[dict[str, Any]], probe_records: list[dict[str, Any]], name: str) -> tuple[list[str], dict[str, Any]]:
    """Ensure structural probe sentences are held out from injection training.

    Exact text overlap is the minimum required gate. Later BabyLM corpus
    transforms can add stronger source-document split checks, but this catches
    the most damaging leakage for the controlled generator.
    """
    train_texts = {str(r.get("text", "")) for r in train_records}
    probe_texts = {str(r.get("text", "")) for r in probe_records}
    overlap = sorted(t for t in (train_texts & probe_texts) if t)
    errors = [f"{name}: train/probe exact text overlap {len(overlap)}; example {overlap[0]!r}"] if overlap else []
    return errors, {"train_probe_exact_overlap": len(overlap)}
