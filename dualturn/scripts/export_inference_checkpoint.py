#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from score_fvad_checkpoint import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_LORA_TARGETS,
    EXPERIMENTS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a compact, self-describing inference checkpoint.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    try:
        source = torch.load(args.input, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        source = torch.load(args.input, map_location="cpu")
    if "model_state" not in source:
        raise KeyError(f"Training checkpoint has no model_state: {args.input}")

    spec = EXPERIMENTS[args.experiment]
    state = {
        key: value
        for key, value in source["model_state"].items()
        if not key.startswith("model._mimi_encoder.")
    }
    removed = len(source["model_state"]) - len(state)
    saved_args = dict(source.get("args") or {})
    saved_args.update({
        "backbone": spec.backbone,
        "train_mode": spec.train_mode,
        "dualturn_fvad_head": spec.fvad_head,
        "fvad_target_scheme": spec.target_scheme,
        "dualturn_losses": spec.losses,
        "dualturn_lora_r": int(saved_args.get("dualturn_lora_r", 16)),
        "dualturn_lora_alpha": int(saved_args.get("dualturn_lora_alpha", 32)),
        "dualturn_lora_dropout": float(saved_args.get("dualturn_lora_dropout", 0.05)),
        "dualturn_lora_targets": saved_args.get("dualturn_lora_targets", DEFAULT_LORA_TARGETS),
        "threshold_ratio": float(saved_args.get("threshold_ratio", 0.5)),
    })
    output = {
        "format": "turn-taking-naturalness-inference-v1",
        "model_state": state,
        "args": saved_args,
        "epoch": source.get("epoch"),
        "global_step": source.get("global_step"),
        "naturalness_metric": source.get("naturalness_metric"),
        "naturalness_metric_value": source.get("naturalness_metric_value"),
        "source_checkpoint": args.input.name,
        "experiment": args.experiment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    summary = {
        "output": str(args.output),
        "experiment": args.experiment,
        "state_tensors": len(state),
        "removed_lazy_mimi_tensors": removed,
        "size_bytes": args.output.stat().st_size,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
