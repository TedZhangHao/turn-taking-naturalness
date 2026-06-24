#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_vap_nll_naturalness import (
    DEFAULT_UNNATURAL,
    add_score_1_5_fields,
    aggregate_unit_rows,
    extract_utterance_units,
    fit_score_1_5_scale,
    frame_slice_for_unit,
    load_silero_models,
    load_stereo_audio,
    natural_reference_from_meta,
    observed_vad_50hz,
    read_json,
    read_manifest,
    sample_stem_from_session,
    scoring_valid_mask,
    summarize_pairs,
    summarize_segments,
    tqdm,
    write_csv,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "dualturn" / "outputs" / "dualturn_fvad_nll_naturalness"
DEFAULT_MODEL_ID = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = 1920  # 24 kHz / 12.5 Hz
CHUNK_FRAMES = 375
FVAD_BIN_EDGES = [3, 6, 12, 25]
TARGET_VAD_SR = 16_000
EPS = 1e-6


def load_audio(path: Path, target_sr: int = 24_000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.float()


def load_dualturn_model(model_id: str, device: torch.device, *, local_files_only: bool) -> torch.nn.Module:
    # SCRIPT_DIR is inserted into sys.path above. Importing through
    # ``dualturn.scripts`` fails because scripts/ is intentionally not a package.
    from train_fvad_head import OfficialDualTurnFVADModel

    model = OfficialDualTurnFVADModel(
        model_id,
        local_files_only=local_files_only,
        head_init="pretrained",
        fvad_head_type="native8",
        multitask=True,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=[],
    )
    model.to(device).eval()
    print(f"Loaded DualTurn checkpoint with restored task-layer attention for FVAD NLL: {model_id}")
    return model


@torch.no_grad()
def run_signal_inference(model: torch.nn.Module, audio_24k: torch.Tensor, device: torch.device) -> dict[str, np.ndarray]:
    preds: dict[str, list[np.ndarray]] = {k: [] for k in ["vad", "fvad", "eot", "hold", "bot", "bc"]}
    chunk_samples = CHUNK_FRAMES * SAMPLES_PER_FRAME
    for start in range(0, audio_24k.shape[-1], chunk_samples):
        chunk = audio_24k[:, start:start + chunk_samples].to(device)
        logits = model.forward_all(chunk)
        for name in preds:
            preds[name].append(torch.sigmoid(logits[name]).float().cpu().numpy()[0])
    return {k: np.concatenate(v, axis=0) for k, v in preds.items()}


def downsample_50hz_to_12_5hz(vad_50hz: np.ndarray, *, threshold: float) -> np.ndarray:
    """Convert 20 ms VAD frames into DualTurn 80 ms frames."""
    vad_50hz = np.asarray(vad_50hz, dtype=np.float32)
    n = vad_50hz.shape[0] // 4
    if n <= 0:
        return np.zeros((0, vad_50hz.shape[1]), dtype=np.float32)
    grouped = vad_50hz[: n * 4].reshape(n, 4, vad_50hz.shape[1]).mean(axis=1)
    return (grouped >= threshold).astype(np.float32)


def compute_fvad_targets_np(vad_12_5hz: np.ndarray, bin_edges: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Mirror dualturn-main.dualturn.model.losses.compute_fvad_targets.

    Bins are future-only:
      edge [3, 6, 12, 25] gives t+1..t+3, t+4..t+6,
      t+7..t+12, t+13..t+25.
    """
    vad = np.asarray(vad_12_5hz, dtype=np.float32)
    if vad.ndim != 2 or vad.shape[1] != 2:
        raise ValueError(f"Expected VAD shape [T, 2], got {vad.shape}")

    T = int(vad.shape[0])
    num_bins = len(bin_edges)
    targets = np.zeros((T, num_bins * 2), dtype=np.float32)
    mask = np.zeros((T,), dtype=bool)
    if T == 0:
        return targets, mask

    max_offset = int(bin_edges[-1])
    valid_T = T - max_offset
    if valid_T <= 0:
        return targets, mask

    arange = np.arange(valid_T)
    cumulative = []
    for ch in range(2):
        padded = np.pad(vad[:, ch], (1, 0), mode="constant")
        cumulative.append(np.cumsum(padded, dtype=np.float64))

    prev_edge = 0
    for bin_idx, edge in enumerate(bin_edges):
        start_off = prev_edge + 1
        bin_size = edge - start_off + 1
        idx_end = arange + edge + 1
        idx_start = arange + start_off
        targets[:valid_T, bin_idx] = (cumulative[0][idx_end] - cumulative[0][idx_start]) / bin_size
        targets[:valid_T, num_bins + bin_idx] = (cumulative[1][idx_end] - cumulative[1][idx_start]) / bin_size
        prev_edge = edge

    mask[:valid_T] = True
    return targets, mask


def fvad_bernoulli_frame_nll(
    fvad_probs: np.ndarray,
    targets: np.ndarray,
    fvad_mask: np.ndarray,
    *,
    head_reduction: str,
) -> dict[str, np.ndarray]:
    n = min(int(fvad_probs.shape[0]), int(targets.shape[0]), int(fvad_mask.shape[0]))
    if n <= 0:
        return {
            "nll": np.zeros((0,), dtype=np.float32),
            "targets": np.zeros((0, 8), dtype=np.float32),
            "probs": np.zeros((0, 8), dtype=np.float32),
            "fvad_valid": np.zeros((0,), dtype=bool),
        }

    probs = np.asarray(fvad_probs[:n], dtype=np.float32)
    targets = np.asarray(targets[:n], dtype=np.float32)
    valid = np.asarray(fvad_mask[:n], dtype=bool)
    probs = np.clip(probs, EPS, 1.0 - EPS)
    per_head = -(targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs))
    if head_reduction == "sum":
        vals = per_head.sum(axis=1)
    elif head_reduction == "mean":
        vals = per_head.mean(axis=1)
    else:
        raise ValueError(f"Unsupported head_reduction: {head_reduction}")

    nll = np.full((n,), np.nan, dtype=np.float32)
    nll[valid] = vals[valid].astype(np.float32)
    return {
        "nll": nll,
        "targets": targets,
        "probs": probs,
        "fvad_valid": valid,
    }


def frame_rows_for_dualturn_segment(
    *,
    segment_id: str,
    condition: str,
    nll: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    fvad_valid: np.ndarray,
    context_s: float,
) -> list[dict[str, Any]]:
    context_valid = scoring_valid_mask(len(nll), FRAME_RATE, context_s)
    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(nll):
        target_vec = targets[idx].tolist() if idx < len(targets) else []
        prob_vec = probs[idx].tolist() if idx < len(probs) else []
        is_fvad_valid = bool(fvad_valid[idx]) if idx < len(fvad_valid) else False
        rows.append({
            "segment_id": segment_id,
            "condition": condition,
            "frame_idx": int(idx),
            "time": float(idx / FRAME_RATE),
            "nll": float(value) if math.isfinite(float(value)) else float("nan"),
            "scoring_valid": bool(context_valid[idx] and is_fvad_valid),
            "fvad_valid": is_fvad_valid,
            "target_fvad": json.dumps(target_vec),
            "pred_fvad": json.dumps(prob_vec),
        })
    return rows


def unit_rows_from_dualturn_units(
    nll: np.ndarray,
    units: list[dict[str, Any]],
    *,
    segment_id: str,
    context_s: float,
    min_unit_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        lo, hi = frame_slice_for_unit(unit, len(nll), FRAME_RATE, context_s)
        if hi <= lo:
            continue
        vals = nll[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size < min_unit_frames:
            continue
        rows.append({
            "segment_id": segment_id,
            "unit_id": len(rows),
            "unit_type": unit["unit_type"],
            "start_time": float(unit["start_time"]),
            "end_time": float(unit["end_time"]),
            "num_frames": int(vals.size),
            "unit_nll": float(vals.mean()),
            "speaker": unit.get("speaker"),
            "partner": unit.get("partner"),
            "detail": json.dumps(unit.get("detail", {}), ensure_ascii=False),
        })

    if rows:
        return rows

    valid = scoring_valid_mask(len(nll), FRAME_RATE, context_s) & np.isfinite(nll)
    vals = nll[valid]
    if vals.size == 0:
        return []
    idx = np.flatnonzero(valid)
    return [{
        "segment_id": segment_id,
        "unit_id": 0,
        "unit_type": "fallback_all_valid",
        "start_time": float(idx[0] / FRAME_RATE),
        "end_time": float((idx[-1] + 1) / FRAME_RATE),
        "num_frames": int(vals.size),
        "unit_nll": float(vals.mean()),
        "speaker": None,
        "partner": None,
        "detail": json.dumps({"reason": "no_valid_utterance_units"}, ensure_ascii=False),
    }]


@torch.no_grad()
def score_audio(
    model: torch.nn.Module,
    audio_path: Path,
    *,
    segment_id: str,
    condition: str,
    device: torch.device,
    silero_models: list[Any] | None,
    vad_source: str,
    rms_threshold: float,
    silero_threshold: float,
    silero_min_speech_ms: int,
    silero_min_silence_ms: int,
    clean_min_speech_ms: int,
    clean_min_silence_ms: int,
    vad_downsample_threshold: float,
    context_s: float,
    tail_gamma: float,
    lam: float,
    min_unit_frames: int,
    unit_pre_s: float,
    unit_post_s: float,
    min_utterance_s: float,
    unit_mode: str,
    utterance_merge_gap_s: float,
    utterance_merge_other_max_ratio: float,
    head_reduction: str,
    save_frame_scores: bool,
) -> dict[str, Any]:
    audio_24k = load_audio(audio_path)
    duration_s = float(audio_24k.shape[-1] / 24_000)
    preds = run_signal_inference(model, audio_24k, device)
    fvad_probs = np.asarray(preds["fvad"], dtype=np.float32)

    audio_16k = load_stereo_audio(audio_path, TARGET_VAD_SR)
    vad_pack_50 = observed_vad_50hz(
        audio_16k,
        sr=TARGET_VAD_SR,
        vad_source=vad_source,
        silero_models=silero_models,
        rms_threshold=rms_threshold,
        silero_threshold=silero_threshold,
        silero_min_speech_ms=silero_min_speech_ms,
        silero_min_silence_ms=silero_min_silence_ms,
        clean_min_speech_ms=clean_min_speech_ms,
        clean_min_silence_ms=clean_min_silence_ms,
    )
    raw_vad_12 = downsample_50hz_to_12_5hz(vad_pack_50["raw"], threshold=vad_downsample_threshold)
    clean_vad_12 = downsample_50hz_to_12_5hz(vad_pack_50["clean"], threshold=vad_downsample_threshold)

    targets, fvad_valid = compute_fvad_targets_np(clean_vad_12, FVAD_BIN_EDGES)
    n = min(int(fvad_probs.shape[0]), int(targets.shape[0]))
    fvad_probs = fvad_probs[:n]
    targets = targets[:n]
    fvad_valid = fvad_valid[:n]
    clean_vad_12 = clean_vad_12[:n]
    raw_vad_12 = raw_vad_12[:n]

    frame_out = fvad_bernoulli_frame_nll(
        fvad_probs,
        targets,
        fvad_valid,
        head_reduction=head_reduction,
    )
    nll = frame_out["nll"]
    units = extract_utterance_units(
        clean_vad_12,
        frame_hz=FRAME_RATE,
        duration_s=duration_s,
        unit_pre_s=unit_pre_s,
        unit_post_s=unit_post_s,
        min_utterance_s=min_utterance_s,
        unit_mode=unit_mode,
        utterance_merge_gap_s=utterance_merge_gap_s,
        utterance_merge_other_max_ratio=utterance_merge_other_max_ratio,
    )
    unit_rows = unit_rows_from_dualturn_units(
        nll,
        units,
        segment_id=segment_id,
        context_s=context_s,
        min_unit_frames=min_unit_frames,
    )
    aggregate = aggregate_unit_rows(unit_rows, gamma=tail_gamma, lam=lam)
    unit_counts = Counter(row["unit_type"] for row in unit_rows)
    aggregate.update({
        "segment_id": segment_id,
        "condition": condition,
        "audio_path": str(audio_path),
        "duration_s": duration_s,
        "vad_source": vad_source,
        "num_nll_frames": int(len(nll)),
        "num_raw_vad_frames": int(raw_vad_12.shape[0]),
        "num_clean_vad_frames": int(clean_vad_12.shape[0]),
        "num_fvad_valid_frames": int(np.asarray(frame_out["fvad_valid"], dtype=bool).sum()),
        "unit_type_counts": dict(sorted(unit_counts.items())),
        "units": unit_rows,
        "frames": frame_rows_for_dualturn_segment(
            segment_id=segment_id,
            condition=condition,
            nll=nll,
            targets=frame_out["targets"],
            probs=frame_out["probs"],
            fvad_valid=frame_out["fvad_valid"],
            context_s=context_s,
        ) if save_frame_scores else [],
    })
    return aggregate


def main() -> None:
    ap = argparse.ArgumentParser(description="DualTurn FVAD Bernoulli NLL utterance-boundary naturalness scorer.")
    ap.add_argument("--unnatural-manifest", type=Path, default=DEFAULT_UNNATURAL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vad-source", choices=["silero", "rms"], default="silero")
    ap.add_argument("--rms-threshold", type=float, default=0.015)
    ap.add_argument("--silero-threshold", type=float, default=0.5)
    ap.add_argument("--silero-min-speech-ms", type=int, default=100)
    ap.add_argument("--silero-min-silence-ms", type=int, default=50)
    ap.add_argument("--clean-min-speech-ms", type=int, default=150)
    ap.add_argument("--clean-min-silence-ms", type=int, default=150)
    ap.add_argument("--vad-downsample-threshold", type=float, default=0.5)
    ap.add_argument("--context-s", type=float, default=3.0)
    ap.add_argument("--tail-gamma", type=float, default=0.25)
    ap.add_argument("--lambda-mean", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None, help="Deprecated alias: tail_gamma = 1 - tau.")
    ap.add_argument(
        "--head-reduction",
        choices=["mean", "sum"],
        default="sum",
        help="Sum gives the factorized 8-bit joint NLL; mean is retained only for legacy runs.",
    )
    ap.add_argument("--min-unit-frames", type=int, default=3)
    ap.add_argument("--unit-pre-s", type=float, default=2.0)
    ap.add_argument("--unit-post-s", type=float, default=0.0)
    ap.add_argument("--min-utterance-s", type=float, default=0.5)
    ap.add_argument("--utterance-merge-gap-s", type=float, default=1.0)
    ap.add_argument("--utterance-merge-other-max-ratio", type=float, default=0.2)
    ap.add_argument("--unit-mode", choices=["boundaries", "spans", "both"], default="boundaries")
    ap.add_argument("--save-frame-scores", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--score-low-quantile", type=float, default=0.05)
    ap.add_argument("--score-high-quantile", type=float, default=0.95)
    ap.add_argument("--score-nll-low", type=float, default=None)
    ap.add_argument("--score-nll-high", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.tau is not None:
        args.tail_gamma = max(0.0, min(1.0, 1.0 - args.tau))

    device = torch.device(args.device)
    model = load_dualturn_model(args.model_id, device, local_files_only=args.local_files_only)
    silero_models = load_silero_models() if args.vad_source == "silero" else None
    rows = read_manifest(args.unnatural_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = args.output_dir / "artifacts" / "dualturn_fvad_nll_naturalness"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    unit_rows_all: list[dict[str, Any]] = []
    frame_rows_all: list[dict[str, Any]] = []

    row_iter = tqdm(
        rows,
        total=len(rows),
        desc="DualTurn FVAD NLL pairs",
        unit="pair",
        dynamic_ncols=True,
    )
    for i, row in enumerate(row_iter, 1):
        meta = read_json(row["json_path"])
        edited_audio = Path(row["audio_path"])
        natural_audio = (
            Path(row["natural_audio_path"])
            if row.get("natural_audio_path")
            else natural_reference_from_meta(meta, row["json_path"])
        )
        session_id = row.get("session_id") or row.get("id") or edited_audio.stem
        pair_id = str(meta.get("source_natural_stem", sample_stem_from_session(session_id)) or session_id)
        edit_type = str(meta.get("augmentation_type", row.get("split", "")))

        edited_segment_id = str(session_id)
        natural_segment_id = f"{pair_id}__natural_ref_for__{session_id}"
        common_score_kwargs = dict(
            device=device,
            silero_models=silero_models,
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
        edited = score_audio(model, edited_audio, segment_id=edited_segment_id, condition="edited", **common_score_kwargs)
        natural = score_audio(model, natural_audio, segment_id=natural_segment_id, condition="natural", **common_score_kwargs)

        for scored, version, condition in [(natural, "original", "natural"), (edited, "edited", "edited")]:
            segment_rows.append({
                "segment_id": scored["segment_id"],
                "condition": condition,
                "pair_id": pair_id,
                "version": version,
                "edit_type": edit_type,
                "audio_path": scored["audio_path"],
                "duration_s": scored["duration_s"],
                "vad_source": scored["vad_source"],
                "mean_nll": scored["mean_nll"],
                "tail_nll": scored["tail_nll"],
                "dialog_nll": scored["dialog_nll"],
                "nat_score": scored["nat_score"],
                "num_units": scored["num_units"],
                "tail_k": scored["tail_k"],
                "unit_type_counts": json.dumps(scored["unit_type_counts"], ensure_ascii=False),
                "num_nll_frames": scored["num_nll_frames"],
                "num_raw_vad_frames": scored["num_raw_vad_frames"],
                "num_clean_vad_frames": scored["num_clean_vad_frames"],
                "num_fvad_valid_frames": scored["num_fvad_valid_frames"],
            })
            unit_rows_all.extend(scored["units"])
            frame_rows_all.extend(scored["frames"])

        delta_nll = float(edited["dialog_nll"]) - float(natural["dialog_nll"])
        delta_score = float(natural["nat_score"]) - float(edited["nat_score"])
        pair_rows.append({
            "pair_id": pair_id,
            "edit_type": edit_type,
            "original_segment_id": natural_segment_id,
            "edited_segment_id": edited_segment_id,
            "original_audio_path": str(natural_audio),
            "edited_audio_path": str(edited_audio),
            "original_mean_nll": natural["mean_nll"],
            "original_tail_nll": natural["tail_nll"],
            "original_dialog_nll": natural["dialog_nll"],
            "edited_mean_nll": edited["mean_nll"],
            "edited_tail_nll": edited["tail_nll"],
            "edited_dialog_nll": edited["dialog_nll"],
            "delta_nll": delta_nll,
            "original_nat_score": natural["nat_score"],
            "edited_nat_score": edited["nat_score"],
            "delta_score": delta_score,
            "edited_more_unnatural": bool(delta_nll > 0),
            "original_num_units": natural["num_units"],
            "edited_num_units": edited["num_units"],
            "original_unit_type_counts": json.dumps(natural["unit_type_counts"], ensure_ascii=False),
            "edited_unit_type_counts": json.dumps(edited["unit_type_counts"], ensure_ascii=False),
        })
        print(
            f"[{i:3d}/{len(rows)}] {session_id} type={edit_type} "
            f"orig_nll={natural['dialog_nll']:.4f} edited_nll={edited['dialog_nll']:.4f} "
            f"delta={delta_nll:.4f} units={natural['num_units']}/{edited['num_units']} "
            f"edited>{'yes' if delta_nll > 0 else 'no'}"
        )

    score_scale = fit_score_1_5_scale(
        segment_rows,
        low_quantile=args.score_low_quantile,
        high_quantile=args.score_high_quantile,
        fixed_low=args.score_nll_low,
        fixed_high=args.score_nll_high,
    )
    add_score_1_5_fields(segment_rows, pair_rows, score_scale)

    summary = summarize_pairs(pair_rows)
    segment_summary = summarize_segments(segment_rows)
    pair_csv = out_dir / "pair_scores.csv"
    segment_csv = out_dir / "segment_scores.csv"
    unit_csv = out_dir / "units.csv"
    frame_csv = out_dir / "frame_scores.csv"
    legacy_csv = out_dir / "dualturn_fvad_nll_scores.csv"

    pair_fields = list(pair_rows[0].keys()) if pair_rows else []
    segment_fields = list(segment_rows[0].keys()) if segment_rows else []
    unit_fields = list(unit_rows_all[0].keys()) if unit_rows_all else ["segment_id", "unit_id", "unit_type", "start_time", "end_time", "num_frames", "unit_nll", "speaker", "partner", "detail"]
    frame_fields = list(frame_rows_all[0].keys()) if frame_rows_all else ["segment_id", "condition", "frame_idx", "time", "nll", "scoring_valid", "fvad_valid", "target_fvad", "pred_fvad"]
    write_csv(pair_csv, pair_rows, pair_fields)
    write_csv(legacy_csv, pair_rows, pair_fields)
    write_csv(segment_csv, segment_rows, segment_fields)
    write_csv(unit_csv, unit_rows_all, unit_fields)
    if args.save_frame_scores:
        write_csv(frame_csv, frame_rows_all, frame_fields)

    payload = {
        "metric": "dualturn_fvad_8head_bernoulli_utterance_boundary_nll_no_anchor",
        "config": {
            "unnatural_manifest": str(args.unnatural_manifest),
            "model_id": args.model_id,
            "local_files_only": args.local_files_only,
            "fvad_bin_edges_frames_12_5hz": FVAD_BIN_EDGES,
            "fvad_bins_seconds": [[1 / FRAME_RATE, 3 / FRAME_RATE], [4 / FRAME_RATE, 6 / FRAME_RATE], [7 / FRAME_RATE, 12 / FRAME_RATE], [13 / FRAME_RATE, 25 / FRAME_RATE]],
            "fvad_target_definition": "Official DualTurn future-only bins from compute_fvad_targets: t+1..t+3, t+4..t+6, t+7..t+12, t+13..t+25 at 12.5 Hz.",
            "frame_nll": f"Bernoulli BCE over 8 independent FVAD heads with {args.head_reduction} reduction.",
            "vad_source": args.vad_source,
            "rms_threshold": args.rms_threshold,
            "silero_threshold": args.silero_threshold,
            "silero_min_speech_ms": args.silero_min_speech_ms,
            "silero_min_silence_ms": args.silero_min_silence_ms,
            "clean_min_speech_ms": args.clean_min_speech_ms,
            "clean_min_silence_ms": args.clean_min_silence_ms,
            "vad_downsample_threshold": args.vad_downsample_threshold,
            "context_s": args.context_s,
            "tail_gamma": args.tail_gamma,
            "lambda_mean": args.lambda_mean,
            "mean_nll_coefficient": args.lambda_mean,
            "tail_nll_coefficient": 1.0 - args.lambda_mean,
            "min_unit_frames": args.min_unit_frames,
            "unit_definition": "Same no-anchor VAD utterance boundary units as the VAP scorer. Default boundary window is [boundary - unit_pre_s, boundary + unit_post_s], with unit_post_s=0 for one-size-fits-all pre-boundary scoring.",
            "unit_mode": args.unit_mode,
            "unit_pre_s": args.unit_pre_s,
            "unit_post_s": args.unit_post_s,
            "min_utterance_s": args.min_utterance_s,
            "utterance_merge_gap_s": args.utterance_merge_gap_s,
            "utterance_merge_other_max_ratio": args.utterance_merge_other_max_ratio,
            "score_1_5_scale": score_scale,
            "no_anchor_guarantee": "Scoring ignores edit_start, anchor_time, and all edit metadata except paired file lookup/type summaries.",
        },
        "summary": summary,
        "segment_summary": segment_summary,
        "output_files": {
            "frame_scores": str(frame_csv) if args.save_frame_scores else None,
            "units": str(unit_csv),
            "segment_scores": str(segment_csv),
            "pair_scores": str(pair_csv),
            "legacy_pair_scores": str(legacy_csv),
        },
        "rows": pair_rows,
    }
    json_path = out_dir / "dualturn_fvad_nll_results.json"
    write_json(json_path, payload)

    print("=" * 90)
    print("No-anchor DualTurn FVAD Bernoulli NLL summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Segment score components")
    print(json.dumps(segment_summary, indent=2, ensure_ascii=False))
    print(f"Saved pair CSV    -> {pair_csv}")
    print(f"Saved segment CSV -> {segment_csv}")
    print(f"Saved units CSV   -> {unit_csv}")
    if args.save_frame_scores:
        print(f"Saved frame CSV   -> {frame_csv}")
    print(f"Saved JSON        -> {json_path}")


if __name__ == "__main__":
    main()
