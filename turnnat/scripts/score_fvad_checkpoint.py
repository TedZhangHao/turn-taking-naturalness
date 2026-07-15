#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAP_ROOT = PROJECT_ROOT / "VAP-main"
for path in (PROJECT_ROOT, VAP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_fvad_head import (  # noqa: E402
    DEFAULT_DUALTURN_MODEL_ID,
    NaturalnessFiveTypeEvaluator,
    OfficialDualTurnFVADModel,
    VAP256Model,
    load_training_checkpoint,
    vap_bin_times_to_frames,
)

DEFAULT_EXPERIMENT = "turnnat-dualturn-fvad256"
DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


@dataclass(frozen=True)
class ExperimentSpec:
    backbone: str
    train_mode: str
    fvad_head: str
    target_scheme: str
    losses: str


EXPERIMENTS = {
    DEFAULT_EXPERIMENT: ExperimentSpec("dualturn", "full", "categorical256", "shared-binary", "all"),
}


def saved(payload: dict, key: str, default=None):
    return payload.get("args", {}).get(key, default)


def is_training_payload(obj: object) -> bool:
    return isinstance(obj, dict) and "model_state" in obj and "args" in obj


def is_legacy_training_payload(obj: object) -> bool:
    return isinstance(obj, dict) and "model_state" in obj and "args" not in obj


def add_legacy_training_args(payload: dict) -> dict:
    """Older local checkpoints saved model_state but not the training args."""
    state = payload.get("model_state", {})
    keys = set(state.keys()) if isinstance(state, dict) else set()
    if any("fvad_head_256" in key for key in keys):
        fvad_head = "categorical256"
        target_scheme = "shared-binary"
    else:
        fvad_head = "native8"
        target_scheme = "native-soft"
    losses = "all" if any(key.startswith("task_layer_weights.") for key in keys) else "fvad"
    payload = dict(payload)
    payload["args"] = {
        "backbone": "dualturn",
        "dualturn_model_id": DEFAULT_DUALTURN_MODEL_ID,
        "dualturn_fvad_head": fvad_head,
        "dualturn_losses": losses,
        "fvad_target_scheme": target_scheme,
        "train_mode": "head",
        "dualturn_lora_r": 16,
        "dualturn_lora_alpha": 32,
        "dualturn_lora_dropout": 0.05,
        "dualturn_lora_targets": [],
        "threshold_ratio": 0.5,
    }
    print(
        "Detected legacy DualTurn training checkpoint without args; "
        f"inferred head={fvad_head}, losses={losses}, target_scheme={target_scheme}"
    )
    return payload


def count_manifest_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def build_training_payload_model(payload: dict, *, local_files_only: bool) -> torch.nn.Module:
    backbone = str(saved(payload, "backbone"))
    model_state = payload["model_state"]
    if backbone == "vap":
        model = VAP256Model(None, hidden_dim=0, dropout=0.0, head_init="random")
        model.target_scheme = "shared-binary"
    elif backbone == "dualturn":
        head_type = saved(payload, "dualturn_fvad_head", None)
        if head_type is None and any(k.startswith("vap_head_256.") for k in model_state):
            head_type = "categorical256"
            payload.setdefault("args", {})["dualturn_fvad_head"] = head_type
            payload["args"].setdefault("fvad_target_scheme", "shared-binary")
            print("Detected legacy DualTurn 256-way head keys; using categorical256 FVAD head")
        head_type = str(head_type or "native8")
        train_mode = str(saved(payload, "train_mode", "head"))
        model = OfficialDualTurnFVADModel(
            str(saved(payload, "dualturn_model_id", DEFAULT_DUALTURN_MODEL_ID)),
            local_files_only=local_files_only,
            head_init="random" if head_type == "categorical256" else "pretrained",
            fvad_head_type=head_type,
            multitask=str(saved(payload, "dualturn_losses", "fvad")) == "all",
            use_lora=train_mode == "adapter",
            lora_r=int(saved(payload, "dualturn_lora_r", 16)),
            lora_alpha=int(saved(payload, "dualturn_lora_alpha", 32)),
            lora_dropout=float(saved(payload, "dualturn_lora_dropout", 0.05)),
            lora_target_modules=list(saved(payload, "dualturn_lora_targets", [])),
        )
        model.target_scheme = str(saved(payload, "fvad_target_scheme", "native-soft"))
    else:
        raise ValueError(f"Unsupported checkpoint backbone: {backbone!r}")

    if (
        backbone == "dualturn"
        and "vap_head_256.net.weight" in model_state
        and "fvad_head_256.weight" not in model_state
    ):
        model_state = dict(model_state)
        model_state["fvad_head_256.weight"] = model_state.pop("vap_head_256.net.weight")
        model_state["fvad_head_256.bias"] = model_state.pop("vap_head_256.net.bias")
        print("Mapped legacy vap_head_256.net.* tensors to fvad_head_256.*")

    model_keys = set(model.state_dict().keys())
    unexpected = sorted(set(model_state.keys()) - model_keys)
    if unexpected:
        ignorable = [k for k in unexpected if "._mimi_encoder." in k or "._mimi_encoder" in k or "_mimi_encoder." in k]
        non_ignorable = [k for k in unexpected if k not in set(ignorable)]
        if non_ignorable:
            raise RuntimeError(
                "Checkpoint has unexpected non-Mimi keys that do not match the current model: "
                + ", ".join(non_ignorable[:20])
            )
        model_state = {k: v for k, v in model_state.items() if k not in set(ignorable)}
        print(f"Ignored {len(ignorable)} checkpoint-only Mimi encoder tensors during load")

    missing = sorted(model_keys - set(model_state.keys()))
    ignorable_missing = [k for k in missing if k.startswith("task_layer_weights.")]
    if ignorable_missing:
        print(f"Allowed {len(ignorable_missing)} missing restored task-layer tensors during load")
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(model_state, strict=True)
    return model


def build_official_vap_model(checkpoint: Path) -> torch.nn.Module:
    model = VAP256Model(checkpoint, hidden_dim=0, dropout=0.0, head_init="pretrained")
    model.target_scheme = "shared-binary"
    return model


def load_model_and_metadata(
    checkpoint: Path,
    *,
    local_files_only: bool,
) -> tuple[torch.nn.Module, dict, str]:
    payload = load_training_checkpoint(checkpoint)
    if is_legacy_training_payload(payload):
        payload = add_legacy_training_args(payload)
    if is_training_payload(payload):
        return build_training_payload_model(payload, local_files_only=local_files_only), payload, str(saved(payload, "backbone"))

    # Official VAP checkpoints are plain state_dict files rather than the
    # train_fvad_head.py payload. They still expose a pretrained 256-way VAP
    # head, so wrap them in the same NaturalnessFiveTypeEvaluator interface.
    pseudo_payload = {
        "args": {
            "backbone": "vap",
            "threshold_ratio": 0.5,
        },
        "epoch": None,
        "global_step": 0,
    }
    print("Detected plain VAP state_dict checkpoint; using official VAP 256-way head")
    return build_official_vap_model(checkpoint), pseudo_payload, "vap"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified future-VAD naturalness scorer for official VAP state_dicts and trained FVAD checkpoints."
    )
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "turnnat-dualturn-fvad256.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "benchmark" / "manifests" / "test2.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eval_turnnat_dualturn_fvad256_benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "auto"], default=DEFAULT_EXPERIMENT, help="Accepted for compatibility; checkpoint metadata determines the model profile.")
    parser.add_argument("--list-experiments", action="store_true", help="List compatibility profile names and exit.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vad-source", choices=["silero", "rms"], default="silero")
    parser.add_argument("--rms-threshold", type=float, default=0.015)
    parser.add_argument("--context-s", type=float, default=3.0)
    parser.add_argument("--unit-pre-s", type=float, default=2.0)
    parser.add_argument("--unit-post-s", type=float, default=0.0)
    parser.add_argument("--min-utterance-s", type=float, default=0.2)
    args = parser.parse_args()

    if args.list_experiments:
        for name, spec in EXPERIMENTS.items():
            print(
                f"{name}\tbackbone={spec.backbone}\ttrain_mode={spec.train_mode}\t"
                f"fvad_head={spec.fvad_head}\ttarget_scheme={spec.target_scheme}\t"
                f"losses={spec.losses}"
            )
        return

    device = torch.device(args.device)
    model, payload, backbone = load_model_and_metadata(
        args.checkpoint,
        local_files_only=args.local_files_only,
    )
    model = model.to(device).eval()
    frame_hz = 50.0 if backbone == "vap" else 12.5
    sample_rate = 16_000 if backbone == "vap" else 24_000
    evaluator = NaturalnessFiveTypeEvaluator(
        manifest=args.manifest,
        output_dir=args.output_dir,
        sample_rate=sample_rate,
        samples_per_frame=int(sample_rate / frame_hz),
        model_frame_hz=frame_hz,
        bin_frames_50hz=vap_bin_times_to_frames([0.2, 0.4, 0.6, 0.8], 50.0),
        threshold_ratio=float(saved(payload, "threshold_ratio", 0.5)),
        bernoulli_head_reduction="sum",
        batch_size=args.batch_size,
        limit=args.limit if args.limit is not None else count_manifest_rows(args.manifest),
        vad_source=args.vad_source,
        rms_threshold=args.rms_threshold,
        silero_threshold=0.5,
        silero_min_speech_ms=100,
        silero_min_silence_ms=50,
        clean_min_speech_ms=150,
        clean_min_silence_ms=150,
        context_s=args.context_s,
        tail_gamma=0.25,
        lambda_mean=0.5,
        unit_pre_s=args.unit_pre_s,
        unit_post_s=args.unit_post_s,
        min_unit_frames=5,
        min_utterance_s=args.min_utterance_s,
        utterance_merge_gap_s=1.0,
        utterance_merge_other_max_ratio=0.2,
        unit_mode="boundaries",
    )
    result = evaluator.run(
        model,
        device=device,
        epoch=payload.get("epoch"),
        global_step=int(payload.get("global_step", 0)),
        use_autocast=device.type == "cuda",
        autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    print(json.dumps(result["summary"]["overall"], indent=2))
    print(f"Saved scores to {result['output_dir']}")


if __name__ == "__main__":
    main()
