#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAP_ROOT = PROJECT_ROOT / "VAP-main"
for path in (PROJECT_ROOT, VAP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_fvad_head import (  # noqa: E402
    DEFAULT_DUALTURN_MODEL_ID,
    OfficialDualTurnFVADModel,
    VAP256Model,
    load_training_checkpoint,
    vap_bin_times_to_frames,
)
from fvad_naturalness_eval import NaturalnessFiveTypeEvaluator  # noqa: E402


DEFAULT_EXPERIMENT = "group4-dualturn-full-all6-fvad256"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / f"{DEFAULT_EXPERIMENT}.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "inference"
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
    "group3-vap-head-shared": ExperimentSpec("vap", "head", "categorical256", "shared-binary", "fvad"),
    "group4-vap-full-shared": ExperimentSpec("vap", "full", "categorical256", "shared-binary", "fvad"),
    "group3-dualturn-head-native": ExperimentSpec("dualturn", "head", "native8", "native-soft", "fvad"),
    "group3-dualturn-head-shared": ExperimentSpec("dualturn", "head", "native8", "shared-binary", "fvad"),
    "group4-dualturn-lora-fvad": ExperimentSpec("dualturn", "adapter", "native8", "native-soft", "fvad"),
    "group4-dualturn-lora-all6": ExperimentSpec("dualturn", "adapter", "native8", "native-soft", "all"),
    "group4-dualturn-full-fvad": ExperimentSpec("dualturn", "full", "native8", "native-soft", "fvad"),
    "group4-dualturn-full-all6": ExperimentSpec("dualturn", "full", "native8", "native-soft", "all"),
    DEFAULT_EXPERIMENT: ExperimentSpec("dualturn", "full", "categorical256", "shared-binary", "all"),
}


def saved(payload: dict, key: str, default=None):
    return payload.get("args", {}).get(key, default)


def checkpoint_spec(payload: dict) -> ExperimentSpec:
    backbone = str(saved(payload, "backbone", ""))
    if not backbone:
        raise ValueError("Checkpoint has no args.backbone metadata; use a checkpoint produced by train_fvad_head.py")
    return ExperimentSpec(
        backbone=backbone,
        train_mode=str(saved(payload, "train_mode", "head")),
        fvad_head=(
            "categorical256" if backbone == "vap"
            else str(saved(payload, "dualturn_fvad_head", "native8"))
        ),
        target_scheme=(
            "shared-binary" if backbone == "vap"
            else str(saved(payload, "fvad_target_scheme", "native-soft"))
        ),
        losses=(
            "fvad" if backbone == "vap"
            else str(saved(payload, "dualturn_losses", "fvad"))
        ),
    )


def resolve_experiment(payload: dict, experiment: str) -> tuple[str, ExperimentSpec]:
    if experiment == "auto":
        actual = checkpoint_spec(payload)
        matches = [name for name, spec in EXPERIMENTS.items() if spec == actual]
        return (matches[0] if matches else "custom"), actual
    expected = EXPERIMENTS[experiment]
    if not saved(payload, "backbone"):
        return experiment, expected
    actual = checkpoint_spec(payload)
    if actual != expected:
        mismatches = {
            key: {"checkpoint": asdict(actual)[key], "experiment": asdict(expected)[key]}
            for key in asdict(expected)
            if asdict(actual)[key] != asdict(expected)[key]
        }
        raise ValueError(
            f"Checkpoint does not match --experiment {experiment}: {json.dumps(mismatches)}. "
            "Choose the matching profile or use --experiment auto."
        )
    return experiment, expected


def build_model(payload: dict, spec: ExperimentSpec, *, local_files_only: bool) -> torch.nn.Module:
    if spec.backbone == "vap":
        model = VAP256Model(None, hidden_dim=0, dropout=0.0, head_init="random")
        model.target_scheme = "shared-binary"
    elif spec.backbone == "dualturn":
        model = OfficialDualTurnFVADModel(
            str(saved(payload, "dualturn_model_id", DEFAULT_DUALTURN_MODEL_ID)),
            local_files_only=local_files_only,
            head_init="random" if spec.fvad_head == "categorical256" else "pretrained",
            fvad_head_type=spec.fvad_head,
            multitask=spec.losses == "all",
            use_lora=spec.train_mode == "adapter",
            lora_r=int(saved(payload, "dualturn_lora_r", 16)),
            lora_alpha=int(saved(payload, "dualturn_lora_alpha", 32)),
            lora_dropout=float(saved(payload, "dualturn_lora_dropout", 0.05)),
            lora_target_modules=list(saved(payload, "dualturn_lora_targets", DEFAULT_LORA_TARGETS)),
        )
        model.target_scheme = spec.target_scheme
    else:
        raise ValueError(f"Unsupported checkpoint backbone: {spec.backbone!r}")

    if any(key.startswith("model._mimi_encoder.") for key in payload["model_state"]):
        # Some mid-training checkpoints were saved after inference lazily attached Mimi.
        model.model._get_mimi()
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a trained FVAD checkpoint on natural/edited pairs.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--experiment",
        choices=[*EXPERIMENTS, "auto"],
        default=DEFAULT_EXPERIMENT,
        help=f"Model profile. Default: {DEFAULT_EXPERIMENT}",
    )
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vad-source", choices=["silero", "rms"], default="silero")
    parser.add_argument("--rms-threshold", type=float, default=0.015)
    parser.add_argument("--context-s", type=float, default=3.0)
    parser.add_argument("--unit-pre-s", type=float, default=2.0)
    parser.add_argument("--unit-post-s", type=float, default=0.0)
    args = parser.parse_args()

    if args.list_experiments:
        for name, spec in EXPERIMENTS.items():
            print(f"{name}: {json.dumps(asdict(spec), sort_keys=True)}")
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Place the shared default checkpoint there "
            "or pass --checkpoint /path/to/checkpoint.pt."
        )
    payload = load_training_checkpoint(args.checkpoint)
    experiment, spec = resolve_experiment(payload, args.experiment)
    device = torch.device(args.device)
    model = build_model(payload, spec, local_files_only=args.local_files_only).to(device).eval()
    frame_hz = 50.0 if spec.backbone == "vap" else 12.5
    sample_rate = 16_000 if spec.backbone == "vap" else 24_000
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
        limit=args.limit,
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
        min_utterance_s=0.5,
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
    output_dir = Path(result["output_dir"])
    inference_config = {
        "experiment": experiment,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "checkpoint_args_present": bool(payload.get("args")),
        "model": asdict(spec),
        "manifest": str(args.manifest.resolve()),
        "frame_nll": (
            "256-state categorical cross-entropy in nats"
            if spec.fvad_head == "categorical256"
            else "sum of 8 Bernoulli cross-entropies in nats"
        ),
    }
    (output_dir / "inference_config.json").write_text(
        json.dumps(inference_config, indent=2), encoding="utf-8"
    )
    overall = result["summary"]["overall"]
    print(f"Experiment: {experiment}")
    print(f"Pairwise accuracy: {overall['pairwise_accuracy']:.4f}")
    print(f"C-index: {overall['c_index']:.4f}")
    print(f"Mean delta NLL: {overall['delta_nll']['mean']:.4f}")
    print(f"Saved scores to {result['output_dir']}")


if __name__ == "__main__":
    main()
