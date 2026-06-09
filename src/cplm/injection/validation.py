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
    for rec in train_records:
        if not rec.get("text"):
            errors.append(f"{rec.get('id', '<missing>')}: missing text")
        for key in ["entity", "discoverer", "year", "town"]:
            if key not in rec:
                errors.append(f"{rec.get('id', '<missing>')}: missing {key}")
    depth_counts = Counter()
    for rec in probe_records:
        if not rec.get("prompt") or not rec.get("target"):
            errors.append(f"{rec.get('id', '<missing>')}: missing prompt or target")
        depth_counts[str(rec.get("depth"))] += 1
        if rec.get("target") in str(rec.get("prompt")):
            errors.append(f"{rec.get('id', '<missing>')}: target leaks in prompt")
    return errors, {"n_train": len(train_records), "n_probe": len(probe_records), "probe_depths": dict(depth_counts)}


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
