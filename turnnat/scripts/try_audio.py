#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
VAP_ROOT = PROJECT_ROOT / "VAP-main"
for path in (PROJECT_ROOT, SCRIPT_DIR, VAP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from score_fvad_checkpoint import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    load_model_and_metadata,
    load_training_checkpoint,
    saved,
)
from train_fvad_head import DEFAULT_DUALTURN_MODEL_ID, NaturalnessFiveTypeEvaluator, vap_bin_times_to_frames  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def scalar_segment_row(row: dict[str, Any]) -> dict[str, Any]:
    unit_counts = row.get("unit_type_counts", {})
    if isinstance(unit_counts, dict):
        unit_counts = json.dumps(unit_counts, ensure_ascii=False, sort_keys=True)
    return {
        "segment_id": row.get("segment_id"),
        "condition": row.get("condition"),
        "audio_path": row.get("audio_path"),
        "duration_s": row.get("duration_s"),
        "vad_source": row.get("vad_source"),
        "mean_nll": row.get("mean_nll"),
        "tail_nll": row.get("tail_nll"),
        "dialog_nll": row.get("dialog_nll"),
        "nat_score": row.get("nat_score"),
        "num_units": row.get("num_units"),
        "tail_k": row.get("tail_k"),
        "unit_type_counts": unit_counts,
        "num_nll_frames": row.get("num_nll_frames"),
        "num_raw_vad_frames": row.get("num_raw_vad_frames"),
        "num_clean_vad_frames": row.get("num_clean_vad_frames"),
        "num_fvad_valid_frames": row.get("num_fvad_valid_frames"),
    }


def load_official_vap(args: argparse.Namespace, device: torch.device):
    from score_vap_nll_naturalness import load_vap_model

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required with --score-backend official-vap")
    return load_vap_model(args.checkpoint, device)


def score_with_official_vap(model, audio_path: Path, *, segment_id: str, condition: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from score_vap_nll_naturalness import load_silero_models, score_audio

    if not hasattr(args, "_silero_models"):
        args._silero_models = load_silero_models() if args.vad_source == "silero" else None
    return score_audio(
        model,
        audio_path,
        segment_id=segment_id,
        condition=condition,
        device=device,
        silero_models=args._silero_models,
        vad_source=args.vad_source,
        rms_threshold=args.rms_threshold,
        silero_threshold=args.silero_threshold,
        silero_min_speech_ms=args.silero_min_speech_ms,
        silero_min_silence_ms=args.silero_min_silence_ms,
        clean_min_speech_ms=args.clean_min_speech_ms,
        clean_min_silence_ms=args.clean_min_silence_ms,
        chunk_s=args.chunk_s,
        context_s=args.context_s,
        tail_gamma=args.tail_gamma,
        lam=args.lambda_mean,
        min_unit_frames=args.min_unit_frames,
        unit_pre_s=args.unit_pre_s,
        unit_post_s=args.unit_post_s,
        min_utterance_s=args.min_utterance_s,
        unit_mode=args.unit_mode,
        utterance_merge_gap_s=args.utterance_merge_gap_s,
        utterance_merge_other_max_ratio=args.utterance_merge_other_max_ratio,
        save_frame_scores=args.save_frame_scores,
    )


def load_official_dualturn(args: argparse.Namespace, device: torch.device):
    from score_dualturn_fvad_nll_naturalness import load_dualturn_model

    return load_dualturn_model(args.model_id, device, local_files_only=args.local_files_only)


def score_with_official_dualturn(model, audio_path: Path, *, segment_id: str, condition: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from score_dualturn_fvad_nll_naturalness import load_silero_models, score_audio

    if not hasattr(args, "_silero_models"):
        args._silero_models = load_silero_models() if args.vad_source == "silero" else None
    return score_audio(
        model,
        audio_path,
        segment_id=segment_id,
        condition=condition,
        device=device,
        silero_models=args._silero_models,
        vad_source=args.vad_source,
        rms_threshold=args.rms_threshold,
        silero_threshold=args.silero_threshold,
        silero_min_speech_ms=args.silero_min_speech_ms,
        silero_min_silence_ms=args.silero_min_silence_ms,
        clean_min_speech_ms=args.clean_min_speech_ms,
        clean_min_silence_ms=args.clean_min_silence_ms,
        vad_downsample_threshold=args.vad_downsample_threshold,
        context_s=args.context_s,
        tail_gamma=args.tail_gamma,
        lam=args.lambda_mean,
        min_unit_frames=args.min_unit_frames,
        unit_pre_s=args.unit_pre_s,
        unit_post_s=args.unit_post_s,
        min_utterance_s=args.min_utterance_s,
        unit_mode=args.unit_mode,
        utterance_merge_gap_s=args.utterance_merge_gap_s,
        utterance_merge_other_max_ratio=args.utterance_merge_other_max_ratio,
        head_reduction=args.head_reduction,
        save_frame_scores=args.save_frame_scores,
    )


def load_fvad_checkpoint(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required with --score-backend fvad-checkpoint")
    model, payload, backbone = load_model_and_metadata(args.checkpoint, local_files_only=args.local_files_only)
    return model.to(device).eval(), payload, backbone


def score_with_fvad_checkpoint(model_pack, audio_path: Path, *, segment_id: str, condition: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model, payload, backbone = model_pack
    frame_hz = 50.0 if backbone == "vap" else 12.5
    sample_rate = 16_000 if backbone == "vap" else 24_000
    dummy_manifest = PROJECT_ROOT / "samples" / "manifest.csv"
    evaluator = NaturalnessFiveTypeEvaluator(
        manifest=dummy_manifest,
        output_dir=args.output_dir,
        sample_rate=sample_rate,
        samples_per_frame=int(sample_rate / frame_hz),
        model_frame_hz=frame_hz,
        bin_frames_50hz=vap_bin_times_to_frames([0.2, 0.4, 0.6, 0.8], 50.0),
        threshold_ratio=float(saved(payload, "threshold_ratio", 0.5)),
        bernoulli_head_reduction=args.head_reduction,
        batch_size=1,
        limit=0,
        vad_source=args.vad_source,
        rms_threshold=args.rms_threshold,
        silero_threshold=args.silero_threshold,
        silero_min_speech_ms=args.silero_min_speech_ms,
        silero_min_silence_ms=args.silero_min_silence_ms,
        clean_min_speech_ms=args.clean_min_speech_ms,
        clean_min_silence_ms=args.clean_min_silence_ms,
        context_s=args.context_s,
        tail_gamma=args.tail_gamma,
        lambda_mean=args.lambda_mean,
        unit_pre_s=args.unit_pre_s,
        unit_post_s=args.unit_post_s,
        min_unit_frames=args.min_unit_frames,
        min_utterance_s=args.min_utterance_s,
        utterance_merge_gap_s=args.utterance_merge_gap_s,
        utterance_merge_other_max_ratio=args.utterance_merge_other_max_ratio,
        unit_mode=args.unit_mode,
    )
    item = {
        "segment_id": segment_id,
        "condition": condition,
        "version": condition,
        "pair_id": "try_audio",
        "edit_type": args.perturbation_type or "input",
        "audio_path": audio_path,
    }
    prepared = [evaluator._prepare_segment(item)]
    scored = evaluator._score_prepared_batch(
        model,
        prepared,
        device=device,
        use_autocast=device.type == "cuda",
        autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )[0]
    scored["vad_source"] = args.vad_source
    return scored


def load_backend(args: argparse.Namespace, device: torch.device):
    if args.score_backend == "official-vap":
        return load_official_vap(args, device), score_with_official_vap
    if args.score_backend == "official-dualturn":
        return load_official_dualturn(args, device), score_with_official_dualturn
    if args.score_backend == "fvad-checkpoint":
        return load_fvad_checkpoint(args, device), score_with_fvad_checkpoint
    raise ValueError(f"Unsupported score backend: {args.score_backend}")


def write_single_or_pair_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], pair_row: dict[str, Any] | None) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_rows = [scalar_segment_row(row) for row in rows]
    write_csv(args.output_dir / "segment_scores.csv", segment_rows)
    unit_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for row in rows:
        unit_rows.extend(row.get("units", []))
        frame_rows.extend(row.get("frames", []))
    if unit_rows:
        write_csv(args.output_dir / "units.csv", unit_rows)
    if frame_rows:
        write_csv(args.output_dir / "frame_scores.csv", frame_rows)
    if pair_row is not None:
        write_csv(args.output_dir / "pair_scores.csv", [pair_row])
    summary = {
        "score_backend": args.score_backend,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "model_id": args.model_id if args.score_backend == "official-dualturn" else None,
        "segment_scores": segment_rows,
        "pair_score": pair_row,
        "metric_note": "Lower DialogNLL means the model finds the audio more natural under future-VAD labels.",
    }
    write_json(args.output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved outputs to {args.output_dir}")


def run_score(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, scorer = load_backend(args, device)
    row = scorer(model, args.audio, segment_id=args.segment_id or args.audio.stem, condition="input", args=args, device=device)
    write_single_or_pair_outputs(args, [row], None)


def run_score_pair(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, scorer = load_backend(args, device)
    pair_id = args.pair_id or args.perturbed_audio.stem
    natural = scorer(model, args.natural_audio, segment_id=f"{pair_id}__natural", condition="natural", args=args, device=device)
    perturbed = scorer(model, args.perturbed_audio, segment_id=f"{pair_id}__perturbed", condition="perturbed", args=args, device=device)
    delta = float(perturbed["dialog_nll"]) - float(natural["dialog_nll"])
    pair_row = {
        "pair_id": pair_id,
        "perturbation_type": args.perturbation_type or "unknown",
        "natural_audio_path": str(args.natural_audio),
        "perturbed_audio_path": str(args.perturbed_audio),
        "natural_dialog_nll": natural["dialog_nll"],
        "perturbed_dialog_nll": perturbed["dialog_nll"],
        "delta_nll": delta,
        "perturbed_more_unnatural": bool(delta > 0),
    }
    write_single_or_pair_outputs(args, [natural, perturbed], pair_row)


def run_perturb_and_score(args: argparse.Namespace) -> None:
    perturb_dir = args.output_dir / "generated"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "turnnat" / "scripts" / "build_perturbations.py"),
        "make-unnatural",
        "--natural-csv", str(args.natural_csv),
        "--out-root", str(perturb_dir),
        "--split", args.split,
        "--per-type", str(args.per_type),
        "--types", args.perturbation_type,
        "--short-context",
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.turn_source:
        cmd.extend(["--turn-source", args.turn_source])
    subprocess.run(cmd, check=True)

    manifest = perturb_dir / "manifests" / f"{args.split}.csv"
    score_dir = args.output_dir / "scores"
    if args.score_backend == "official-vap":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required to score generated perturbations with official-vap")
        score_cmd = [
            sys.executable, str(PROJECT_ROOT / "turnnat" / "scripts" / "score_vap_nll_naturalness.py"),
            "--unnatural-manifest", str(manifest),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(score_dir),
            "--device", args.device,
            "--vad-source", args.vad_source,
            "--unit-pre-s", str(args.unit_pre_s),
            "--unit-post-s", str(args.unit_post_s),
            "--min-utterance-s", str(args.min_utterance_s),
        ]
    elif args.score_backend == "official-dualturn":
        score_cmd = [
            sys.executable, str(PROJECT_ROOT / "turnnat" / "scripts" / "score_dualturn_fvad_nll_naturalness.py"),
            "--unnatural-manifest", str(manifest),
            "--output-dir", str(score_dir),
            "--model-id", args.model_id,
            "--device", args.device,
            "--vad-source", args.vad_source,
            "--unit-pre-s", str(args.unit_pre_s),
            "--unit-post-s", str(args.unit_post_s),
            "--min-utterance-s", str(args.min_utterance_s),
        ]
        score_cmd.append("--local-files-only" if args.local_files_only else "--no-local-files-only")
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required to score generated perturbations with fvad-checkpoint")
        score_cmd = [
            sys.executable, str(PROJECT_ROOT / "turnnat" / "scripts" / "score_fvad_checkpoint.py"),
            "--checkpoint", str(args.checkpoint),
            "--experiment", args.experiment,
            "--manifest", str(manifest),
            "--output-dir", str(score_dir),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--vad-source", args.vad_source,
            "--unit-pre-s", str(args.unit_pre_s),
            "--unit-post-s", str(args.unit_post_s),
        ]
        score_cmd.append("--local-files-only" if args.local_files_only else "--no-local-files-only")
    subprocess.run(score_cmd, check=True)
    print(f"Generated manifest: {manifest}")
    print(f"Score output root: {score_dir}")


def add_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--score-backend", choices=["official-dualturn", "official-vap", "fvad-checkpoint"], default="official-dualturn")
    parser.add_argument("--checkpoint", type=Path, default=None, help="VAP state dict or released FVAD checkpoint path, depending on --score-backend.")
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "auto"], default="auto", help="FVAD checkpoint profile; use auto for released checkpoints with metadata.")
    parser.add_argument("--batch-size", type=int, default=16, help="Pair batch size used by perturb-and-score with --score-backend fvad-checkpoint.")
    parser.add_argument("--model-id", default=DEFAULT_DUALTURN_MODEL_ID, help="Hugging Face model id for official DualTurn scoring.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vad-source", choices=["silero", "rms"], default="silero")
    parser.add_argument("--rms-threshold", type=float, default=0.015)
    parser.add_argument("--silero-threshold", type=float, default=0.5)
    parser.add_argument("--silero-min-speech-ms", type=int, default=100)
    parser.add_argument("--silero-min-silence-ms", type=int, default=50)
    parser.add_argument("--clean-min-speech-ms", type=int, default=150)
    parser.add_argument("--clean-min-silence-ms", type=int, default=150)
    parser.add_argument("--vad-downsample-threshold", type=float, default=0.5)
    parser.add_argument("--chunk-s", type=float, default=60.0)
    parser.add_argument("--context-s", type=float, default=3.0)
    parser.add_argument("--tail-gamma", type=float, default=0.25)
    parser.add_argument("--lambda-mean", type=float, default=0.5)
    parser.add_argument("--head-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--min-unit-frames", type=int, default=5)
    parser.add_argument("--unit-pre-s", type=float, default=2.0)
    parser.add_argument("--unit-post-s", type=float, default=0.0)
    parser.add_argument("--min-utterance-s", type=float, default=0.5)
    parser.add_argument("--utterance-merge-gap-s", type=float, default=1.0)
    parser.add_argument("--utterance-merge-other-max-ratio", type=float, default=0.2)
    parser.add_argument("--unit-mode", choices=["boundaries", "spans", "both"], default="boundaries")
    parser.add_argument("--save-frame-scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--perturbation-type", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "try_audio")


def main() -> None:
    parser = argparse.ArgumentParser(description="Try Turn-Taking Naturalness scoring and perturbation generation on your own stereo audio.")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score one 2-channel audio file.")
    score.add_argument("--audio", type=Path, required=True)
    score.add_argument("--segment-id", default=None)
    add_score_args(score)
    score.set_defaults(func=run_score)

    pair = sub.add_parser("score-pair", help="Score and compare a natural/perturbed 2-channel audio pair.")
    pair.add_argument("--natural-audio", type=Path, required=True)
    pair.add_argument("--perturbed-audio", type=Path, required=True)
    pair.add_argument("--pair-id", default=None)
    add_score_args(pair)
    pair.set_defaults(func=run_score_pair)

    perturb = sub.add_parser("perturb-and-score", help="Generate perturbations from a natural manifest, then score the generated pairs.")
    perturb.add_argument("--natural-csv", type=Path, required=True, help="Natural manifest with participant audio and metadata/transcripts when available.")
    perturb.add_argument("--split", default="test")
    perturb.add_argument("--per-type", type=int, default=1)
    perturb.add_argument("--seed", type=int, default=None)
    perturb.add_argument("--turn-source", choices=["silero", "rms", "metadata", "metadata_vad"], default="silero")
    add_score_args(perturb)
    perturb.set_defaults(func=run_perturb_and_score)

    args = parser.parse_args()
    if args.perturbation_type is None:
        args.perturbation_type = "late_response" if args.command == "perturb-and-score" else "input"
    if args.command == "perturb-and-score" and args.perturbation_type == "input":
        raise ValueError("perturb-and-score requires --perturbation-type, e.g. late_response")
    args.func(args)


if __name__ == "__main__":
    main()
