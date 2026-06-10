from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import RunSpec, config_hash, save_yaml


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out
    except Exception:
        return None


class ResultManager:
    def __init__(self, repo_root: Path, results_root: Path, spec: RunSpec, config: dict[str, Any]):
        self.repo_root = repo_root
        self.results_root = results_root
        self.spec = spec
        self.config = config
        self.run_dir = results_root / "raw" / spec.stage / spec.group_id / spec.run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.logs_dir = results_root / "logs" / spec.stage
        self.manifests_dir = results_root / "manifests" / spec.stage
        self.index_path = results_root / "indices" / "runs.csv"

    def prepare(self, overwrite: bool = False) -> None:
        if self.run_dir.exists() and not overwrite:
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(self.config, self.run_dir / "config_resolved.yaml")
        manifest = self._manifest()
        self._write_json(self.run_dir / "manifest.json", manifest)
        self._write_json(self.manifests_dir / f"{self.spec.run_id}.json", manifest)
        self._append_index(manifest)

    def _manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.spec.run_id,
            "stage": self.spec.stage,
            "group_id": self.spec.group_id,
            "schedule": self.spec.schedule,
            "onset_fraction": self.spec.onset_fraction,
            "onset_step": self.spec.onset_step,
            "seed": self.spec.seed,
            "readout": self.spec.readout,
            "created_unix": time.time(),
            "config_hash": config_hash(self.config),
            "repo_git_sha": _git_sha(self.repo_root),
            "host": platform.node(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    def _append_index(self, manifest: dict[str, Any]) -> None:
        fieldnames = [
            "run_id",
            "stage",
            "group_id",
            "schedule",
            "onset_fraction",
            "onset_step",
            "seed",
            "readout",
            "created_unix",
            "config_hash",
            "repo_git_sha",
        ]
        exists = self.index_path.exists()
        with open(self.index_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({k: manifest.get(k) for k in fieldnames})

    def append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.run_dir / filename
        with open(path, "a", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
            f.write("\n")

    def save_checkpoint(self, step: int, payload: dict[str, Any]) -> Path:
        import torch

        path = self.ckpt_dir / f"step_{step:08d}.pt"
        torch.save(payload, path)
        return path
