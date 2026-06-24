#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "dualturn" / "scripts" / "train_fvad_head.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a reproducible FVAD experiment YAML.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_config = payload.get("base_config")
    if not base_config:
        raise ValueError("Experiment YAML must define base_config")

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(base_config)]
    for key, value in payload.get("arguments", {}).items():
        if value is None or value is False:
            continue
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            command.append(flag)
        elif isinstance(value, list):
            command.extend([flag, ",".join(str(item) for item in value)])
        else:
            command.extend([flag, str(value)])

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in payload.get("environment", {}).items()})
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
