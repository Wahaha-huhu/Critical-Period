#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo / "scripts" / "run_stage0_grid.py"), "--config", str(repo / args.config), "--overwrite"]
    subprocess.check_call(cmd, cwd=repo)
    # Find latest smoke group and summarize it.
    raw_root = repo / "results" / "raw" / "smoke"
    groups = sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    latest = groups[-1]
    subprocess.check_call([sys.executable, str(repo / "scripts" / "summarize_stage0.py"), "--results", str(latest)], cwd=repo)
    print(f"Smoke test completed: {latest}")


if __name__ == "__main__":
    main()
