from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def slug_float(x: float) -> str:
    text = f"{x:g}"
    return text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class RunSpec:
    stage: str
    group_id: str
    schedule: str
    onset_fraction: float
    onset_step: int
    seed: int
    readout: str = "fixed_age"

    @property
    def run_id(self) -> str:
        return (
            f"{self.stage}__{self.schedule}__onset-{slug_float(self.onset_fraction)}"
            f"__seed-{self.seed}__{self.readout}"
        )


def round_onset(total_steps: int, onset_fraction: float, mode: str = "nearest") -> int:
    raw = onset_fraction * total_steps
    if mode == "nearest":
        step = int(round(raw))
    elif mode == "floor":
        step = int(raw // 1)
    elif mode == "ceil":
        import math

        step = int(math.ceil(raw))
    else:
        raise ValueError(f"Unknown onset rounding mode: {mode}")
    return max(0, min(total_steps, step))


def auto_checkpoint_steps(total_steps: int) -> list[int]:
    steps: set[int] = set(range(0, min(total_steps, 100) + 1))
    if total_steps > 100:
        import math

        # about 40 checkpoints per decade after step 100
        start_log = math.log10(100)
        end_log = math.log10(total_steps)
        n = max(1, int((end_log - start_log) * 40))
        for i in range(n + 1):
            val = round(10 ** (start_log + (end_log - start_log) * i / n))
            if 0 <= val <= total_steps:
                steps.add(int(val))
    steps.add(total_steps)
    return sorted(steps)
