#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress bars are optional.
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


REPO_ROOT = Path(__file__).resolve().parents[2]
VAP_ROOT = REPO_ROOT / "VAP-main"
if str(VAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VAP_ROOT))


DEFAULT_UNNATURAL = REPO_ROOT / "dataset" / "manifests" / "unnatural_roomtone_normalized_45s_1.csv"
DEFAULT_OUT = REPO_ROOT / "turnnat" / "outputs" / "vap_nll_naturalness"
DEFAULT_CKPT = REPO_ROOT / "VAP-main" / "example" / "checkpoints" / "VAP_state_dict.pt"
VAP_STATE_DICT_URL = "https://github.com/ErikEkstedt/VAP/raw/main/example/checkpoints/VAP_state_dict.pt"
FRAME_HZ = 50.0


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_vap_model(ckpt: Path, device: torch.device):
    from vap.modules.VAP import VAP
    from vap.modules.encoder import EncoderCPC
    from vap.modules.modules import TransformerStereo

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing VAP checkpoint: {ckpt}. Download it with:\n"
            f"  mkdir -p {ckpt.parent}\n"
            f"  curl -L {VAP_STATE_DICT_URL} -o {ckpt}"
        )
    model = VAP(EncoderCPC(), TransformerStereo())
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded VAP checkpoint: {ckpt}")
    return model


def load_silero_models() -> list[Any]:
    from silero_vad import load_silero_vad

    print("Loading Silero VAD models for observed VAD labels...")
    return [load_silero_vad(), load_silero_vad()]


def load_stereo_audio(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.float()


def rms_vad_50hz(audio: torch.Tensor, sr: int, threshold: float) -> np.ndarray:
    hop = int(round(sr / FRAME_HZ))
    n_frames = int(math.ceil(audio.shape[-1] / hop))
    pad = n_frames * hop - audio.shape[-1]
    if pad > 0:
        audio = F.pad(audio, (0, pad))
    frames = audio.unfold(-1, hop, hop)
    rms = torch.sqrt(torch.clamp(frames.pow(2).mean(dim=-1), min=1e-10))
    return (rms.transpose(0, 1) >= threshold).cpu().numpy().astype(np.int8)


def silero_vad_per_frame(
    waveform_16k: np.ndarray,
    vad_model: Any,
    *,
    sr: int = 16000,
    frame_ms: int = 20,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 50,
) -> np.ndarray:
    """DualTurn official-style Silero VAD: timestamps -> 50Hz binary VAD."""
    from silero_vad import get_speech_timestamps

    wav_tensor = torch.from_numpy(waveform_16k).float().cpu()
    timestamps = get_speech_timestamps(
        wav_tensor,
        vad_model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    frame_samples = int(frame_ms / 1000 * sr)
    n_frames = len(waveform_16k) // frame_samples
    vad = np.zeros(n_frames, dtype=np.int8)
    for ts in timestamps:
        start_frame = int(ts["start"] // frame_samples)
        end_frame = min(int(ts["end"] // frame_samples), n_frames)
        vad[start_frame:end_frame] = 1
    return vad


def dualturn_clean_vad_1d(
    vad: np.ndarray,
    *,
    frame_hz: float = FRAME_HZ,
    min_speech_ms: int = 150,
    min_silence_ms: int = 150,
) -> np.ndarray:
    """Mirror DualTurn data/process_otospeech.py clean_vad on one channel."""
    out = np.asarray(vad, dtype=np.int8).copy()
    n = len(out)
    frame_ms = 1000.0 / frame_hz
    min_speech = int(min_speech_ms / frame_ms)
    min_silence = int(min_silence_ms / frame_ms)

    in_speech = False
    start = 0
    for i in range(n):
        if out[i] == 1 and not in_speech:
            start = i
            in_speech = True
        elif out[i] == 0 and in_speech:
            if (i - start) < min_speech:
                out[start:i] = 0
            in_speech = False

    in_silence = False
    start = 0
    for i in range(n):
        if out[i] == 0 and not in_silence:
            start = i
            in_silence = True
        elif out[i] == 1 and in_silence:
            if (i - start) < min_silence:
                out[start:i] = 1
            in_silence = False
    return out


def clean_vad(
    vad: np.ndarray,
    *,
    frame_hz: float = FRAME_HZ,
    min_speech_ms: int = 150,
    min_silence_ms: int = 150,
) -> np.ndarray:
    vad = np.asarray(vad, dtype=np.int8)
    out = np.zeros_like(vad, dtype=np.int8)
    for speaker in range(vad.shape[1]):
        out[:, speaker] = dualturn_clean_vad_1d(
            vad[:, speaker],
            frame_hz=frame_hz,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
        )
    return out


def observed_vad_50hz(
    audio: torch.Tensor,
    *,
    sr: int,
    vad_source: str,
    silero_models: list[Any] | None,
    rms_threshold: float,
    silero_threshold: float,
    silero_min_speech_ms: int,
    silero_min_silence_ms: int,
    clean_min_speech_ms: int,
    clean_min_silence_ms: int,
) -> dict[str, np.ndarray]:
    if vad_source == "silero":
        if silero_models is None:
            raise RuntimeError("vad_source='silero' requires loaded Silero models.")
        raw_channels = []
        for ch in range(2):
            raw_channels.append(
                silero_vad_per_frame(
                    audio[ch].detach().cpu().numpy(),
                    silero_models[ch],
                    sr=sr,
                    threshold=silero_threshold,
                    min_speech_duration_ms=silero_min_speech_ms,
                    min_silence_duration_ms=silero_min_silence_ms,
                )
            )
        n = min(len(raw_channels[0]), len(raw_channels[1]))
        raw = np.stack([raw_channels[0][:n], raw_channels[1][:n]], axis=1).astype(np.int8)
    elif vad_source == "rms":
        raw = rms_vad_50hz(audio, sr, rms_threshold).astype(np.int8)
    else:
        raise ValueError(f"Unsupported VAD source: {vad_source}")

    cleaned = clean_vad(
        raw,
        frame_hz=FRAME_HZ,
        min_speech_ms=clean_min_speech_ms,
        min_silence_ms=clean_min_silence_ms,
    )
    return {"raw": raw, "clean": cleaned}


def boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def speech_segments(vad: np.ndarray, frame_hz: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for speaker in range(vad.shape[1]):
        for start, end in boolean_runs(vad[:, speaker] > 0):
            segments.append({
                "speaker": int(speaker),
                "start_frame": int(start),
                "end_frame": int(end),
                "start": float(start / frame_hz),
                "end": float(end / frame_hz),
                "duration": float((end - start) / frame_hz),
            })
    return sorted(segments, key=lambda x: (x["start"], x["end"], x["speaker"]))


def active_ratio(vad: np.ndarray, speaker: int, start_s: float, end_s: float, frame_hz: float) -> float:
    lo = max(0, int(math.floor(start_s * frame_hz)))
    hi = min(vad.shape[0], int(math.ceil(end_s * frame_hz)))
    if hi <= lo:
        return 0.0
    return float((vad[lo:hi, speaker] > 0).mean())


def merge_utterance_segments(
    segments: list[dict[str, Any]],
    vad: np.ndarray,
    *,
    frame_hz: float,
    max_gap_s: float,
    other_max_ratio: float,
) -> list[dict[str, Any]]:
    """Merge same-speaker VAD islands into broader utterances/IPUs.

    Two adjacent islands from the same speaker are merged if the pause between
    them is short enough and the other speaker is mostly silent during that
    pause. This avoids treating within-utterance micro-pauses as separate units.
    """
    merged_all: list[dict[str, Any]] = []
    speakers = sorted({int(seg["speaker"]) for seg in segments})
    for speaker in speakers:
        speaker_segments = sorted(
            [dict(seg) for seg in segments if int(seg["speaker"]) == speaker],
            key=lambda x: (x["start"], x["end"]),
        )
        current: dict[str, Any] | None = None
        for seg in speaker_segments:
            seg["merged_from"] = int(seg.get("merged_from", 1))
            seg["internal_gap_s"] = float(seg.get("internal_gap_s", 0.0))
            if current is None:
                current = seg
                continue
            gap_s = float(seg["start"] - current["end"])
            other = 1 - speaker
            other_ratio = active_ratio(vad, other, current["end"], seg["start"], frame_hz)
            if gap_s <= max_gap_s and other_ratio <= other_max_ratio:
                current["end_frame"] = int(seg["end_frame"])
                current["end"] = float(seg["end"])
                current["duration"] = float(current["end"] - current["start"])
                current["merged_from"] = int(current.get("merged_from", 1)) + int(seg.get("merged_from", 1))
                current["internal_gap_s"] = float(current.get("internal_gap_s", 0.0)) + max(0.0, gap_s)
                current["max_internal_other_ratio"] = max(
                    float(current.get("max_internal_other_ratio", 0.0)),
                    float(other_ratio),
                )
            else:
                merged_all.append(current)
                current = seg
        if current is not None:
            merged_all.append(current)
    return sorted(merged_all, key=lambda x: (x["start"], x["end"], x["speaker"]))


def clamp_unit(start_s: float, end_s: float, duration_s: float) -> tuple[float, float]:
    return max(0.0, float(start_s)), min(float(duration_s), float(end_s))


def add_unit(
    units: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    *,
    unit_type: str,
    start_s: float,
    end_s: float,
    duration_s: float,
    speaker: int,
    detail: dict[str, Any],
) -> None:
    start_s, end_s = clamp_unit(start_s, end_s, duration_s)
    if end_s <= start_s:
        return
    key = (unit_type, round(start_s, 2), round(end_s, 2), speaker)
    if key in seen:
        return
    seen.add(key)
    units.append({
        "unit_type": unit_type,
        "start_time": start_s,
        "end_time": end_s,
        "speaker": speaker,
        "partner": 1 - speaker,
        "detail": detail,
    })


def extract_utterance_units(
    vad: np.ndarray,
    *,
    frame_hz: float,
    duration_s: float,
    unit_pre_s: float,
    unit_post_s: float,
    min_utterance_s: float,
    unit_mode: str,
    utterance_merge_gap_s: float,
    utterance_merge_other_max_ratio: float,
) -> list[dict[str, Any]]:
    """General no-anchor units from VAD utterance boundaries.

    Default unit_mode='boundaries' creates two units per VAD utterance.
    With the default unit_post_s=0, each boundary unit covers [boundary-pre, boundary].
    This scores the model's pre-boundary expectation while keeping the 2s VAP
    future horizon unchanged.
    """
    units: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    raw_segments = speech_segments(vad, frame_hz)
    segments = merge_utterance_segments(
        raw_segments,
        vad,
        frame_hz=frame_hz,
        max_gap_s=utterance_merge_gap_s,
        other_max_ratio=utterance_merge_other_max_ratio,
    )
    for seg in segments:
        if seg["duration"] < min_utterance_s:
            continue
        speaker = int(seg["speaker"])
        detail = {
            "utterance_start_s": seg["start"],
            "utterance_end_s": seg["end"],
            "utterance_duration_s": seg["duration"],
            "merged_from_vad_islands": int(seg.get("merged_from", 1)),
            "internal_gap_s": float(seg.get("internal_gap_s", 0.0)),
            "utterance_merge_gap_s": utterance_merge_gap_s,
            "utterance_merge_other_max_ratio": utterance_merge_other_max_ratio,
        }
        if unit_mode in {"boundaries", "both"}:
            add_unit(
                units,
                seen,
                unit_type="utterance_start",
                start_s=seg["start"] - unit_pre_s,
                end_s=seg["start"] + unit_post_s,
                duration_s=duration_s,
                speaker=speaker,
                detail={**detail, "boundary": "start"},
            )
            add_unit(
                units,
                seen,
                unit_type="utterance_end",
                start_s=seg["end"] - unit_pre_s,
                end_s=seg["end"] + unit_post_s,
                duration_s=duration_s,
                speaker=speaker,
                detail={**detail, "boundary": "end"},
            )
        if unit_mode in {"spans", "both"}:
            add_unit(
                units,
                seen,
                unit_type="utterance_span",
                start_s=seg["start"] - unit_pre_s,
                end_s=seg["end"] + unit_post_s,
                duration_s=duration_s,
                speaker=speaker,
                detail={**detail, "boundary": "span"},
            )
    return sorted(units, key=lambda x: (x["start_time"], x["end_time"], x["unit_type"], x["speaker"]))




def _finite_ms(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _source_ms(edit: dict[str, Any], key: str) -> float | None:
    value = _finite_ms(edit.get(f"source_timeline_{key}"))
    if value is not None:
        return value
    return _finite_ms(edit.get(key))


def _add_boundary(boundaries: list[dict[str, Any]], *, key: str, ms: float | None, condition: str) -> None:
    if ms is None or not math.isfinite(ms):
        return
    boundaries.append({"key": key, "time_s": float(ms) / 1000.0, "condition": condition})


def protected_boundaries_from_meta(meta: dict[str, Any], *, condition: str) -> list[dict[str, Any]]:
    edit_meta = meta.get("edit_meta", {}) if isinstance(meta, dict) else {}
    edits = edit_meta.get("edits", []) if isinstance(edit_meta, dict) else []
    if not isinstance(edits, list):
        return []
    source_crop_start = _finite_ms((edit_meta.get("short_context", {}) or {}).get("source_crop_start_ms")) or 0.0
    out: list[dict[str, Any]] = []

    def edited(edit: dict[str, Any], key: str) -> float | None:
        return _finite_ms(edit.get(key))

    def natural(edit: dict[str, Any], key: str) -> float | None:
        value = _source_ms(edit, key)
        return None if value is None else value - source_crop_start

    def nested_ms(item: dict[str, Any], key: str) -> float | None:
        if condition == "edited":
            return _finite_ms(item.get(key))
        value = _finite_ms(item.get(f"source_timeline_{key}"))
        if value is None:
            value = _finite_ms(item.get(key))
        return None if value is None else value - source_crop_start

    def add_nested_boundaries(edit: dict[str, Any], list_key: str, prefix: str, keys: list[str]) -> None:
        items = edit.get(list_key)
        if not isinstance(items, list):
            return
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for key in keys:
                ms = nested_ms(item, key)
                boundary_key = f"{prefix}_{idx}_{key}"
                _add_boundary(out, key=boundary_key, ms=ms, condition=condition)

    for edit in edits:
        if not isinstance(edit, dict):
            continue
        typ = str(edit.get("edit_type") or "")
        getter = edited if condition == "edited" else natural
        keys: list[str] = []
        if typ == "late_response":
            if condition == "edited":
                keys = [
                    "speaker_start_ms", "speaker_end_ms",
                    "responder_new_start_ms", "responder_new_end_ms",
                    "next_original_speaker_start_ms", "next_original_speaker_end_ms",
                ]
            else:
                keys = [
                    "speaker_start_ms", "speaker_end_ms",
                    "responder_original_start_ms", "responder_original_end_ms",
                    "next_original_speaker_start_ms", "next_original_speaker_end_ms",
                ]
                # next_original_speaker_* is stored on the edited timeline for late_response.
                delay = _finite_ms(edit.get("delay_ms")) or 0.0
                for key in keys:
                    value = natural(edit, key)
                    if value is not None and key.startswith("next_original_speaker_"):
                        value -= delay
                    _add_boundary(out, key=key, ms=value, condition=condition)
                continue
        elif typ in {"early_entry", "early_interruption", "interruption"}:
            if condition == "edited":
                keys = ["speaker_start_ms", "speaker_end_ms", "responder_new_start_ms", "responder_new_end_ms", "next_turn_start_ms", "next_turn_end_ms"]
            else:
                keys = ["speaker_start_ms", "speaker_end_ms", "responder_original_start_ms", "responder_original_end_ms", "next_turn_start_ms", "next_turn_end_ms"]
                shift = _finite_ms(edit.get("global_timeline_shift_ms")) or _finite_ms(edit.get("advance_ms")) or 0.0
                for key in keys:
                    value = natural(edit, key)
                    if value is not None and key.startswith("next_turn_"):
                        value += shift
                    _add_boundary(out, key=key, ms=value, condition=condition)
                continue
        elif typ == "hold_instead_of_shift":
            keys = ["speaker_start_ms", "speaker_end_ms", "removed_responder_start_ms", "removed_responder_end_ms", "next_original_speaker_start_ms", "next_original_speaker_end_ms"]
        elif typ == "shift_instead_of_hold":
            keys = ["hold_start_ms", "hold_end_ms", "insert_ms", "next_hold_start_ms", "next_hold_end_ms"]
        elif typ == "missed_backchannel":
            keys = ["speaker_start_ms", "speaker_end_ms", "next_turn_start_ms", "next_turn_end_ms"]
            if not isinstance(edit.get("removed_backchannels"), list):
                keys.extend(["backchannel_start_ms", "backchannel_end_ms"])
            add_nested_boundaries(edit, "removed_backchannels", "removed_backchannel", ["start_ms", "end_ms"])
        elif typ == "excessive_backchannel":
            keys = ["speaker_start_ms", "speaker_end_ms", "next_turn_start_ms", "next_turn_end_ms"]
            if not isinstance(edit.get("inserted_backchannels"), list):
                keys.extend(["inserted_backchannel_start_ms", "inserted_backchannel_end_ms"])
            add_nested_boundaries(edit, "inserted_backchannels", "inserted_backchannel", ["start_ms", "end_ms"])
        else:
            keys = ["anchor_ms"]
        for key in keys:
            _add_boundary(out, key=key, ms=getter(edit, key), condition=condition)
    return sorted(out, key=lambda x: (x["time_s"], x["key"]))


def protected_units_from_meta(
    meta: dict[str, Any],
    *,
    condition: str,
    duration_s: float,
    unit_pre_s: float,
    unit_post_s: float,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for boundary in protected_boundaries_from_meta(meta, condition=condition):
        key = str(boundary["key"])
        t = float(boundary["time_s"])
        add_unit(
            units,
            seen,
            unit_type=f"protected_{key.replace('_ms', '')}",
            start_s=t - unit_pre_s,
            end_s=t + unit_post_s,
            duration_s=duration_s,
            speaker=-1,
            detail={"boundary_key": key, "boundary_time_s": t, "unit_source": "protected_metadata"},
        )
    return sorted(units, key=lambda x: (x["start_time"], x["end_time"], x["unit_type"]))


def scoring_valid_mask(nll_len: int, frame_hz: float, context_s: float) -> np.ndarray:
    mask = np.ones(nll_len, dtype=bool)
    context_frames = int(math.ceil(context_s * frame_hz))
    mask[: min(context_frames, nll_len)] = False
    return mask


def fallback_boundary_mask(vad: np.ndarray, nll_len: int, frame_hz: float, context_s: float, window_s: float = 2.0) -> np.ndarray:
    n = min(nll_len, vad.shape[0])
    mask = np.zeros(nll_len, dtype=bool)
    if n <= 1:
        return mask
    window = int(round(window_s * frame_hz))
    changes = np.flatnonzero(np.any(vad[1:n] != vad[: n - 1], axis=1)) + 1
    for idx in changes:
        lo = max(0, int(idx) - window)
        hi = min(nll_len, int(idx) + window + 1)
        mask[lo:hi] = True
    return mask & scoring_valid_mask(nll_len, frame_hz, context_s)


def frame_slice_for_unit(unit: dict[str, Any], nll_len: int, frame_hz: float, context_s: float) -> tuple[int, int]:
    valid_start = int(math.ceil(context_s * frame_hz))
    lo = max(valid_start, int(math.floor(float(unit["start_time"]) * frame_hz)))
    hi = min(nll_len, int(math.ceil(float(unit["end_time"]) * frame_hz)))
    return lo, hi


def unit_rows_from_units(
    nll: np.ndarray,
    units: list[dict[str, Any]],
    *,
    vad: np.ndarray,
    segment_id: str,
    frame_hz: float,
    context_s: float,
    min_unit_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in units:
        lo, hi = frame_slice_for_unit(unit, len(nll), frame_hz, context_s)
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

    boundary_mask = fallback_boundary_mask(vad, len(nll), frame_hz, context_s)
    if int(boundary_mask.sum()) >= min_unit_frames:
        idx = np.flatnonzero(boundary_mask)
        vals = nll[boundary_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size >= min_unit_frames:
            return [{
                "segment_id": segment_id,
                "unit_id": 0,
                "unit_type": "fallback_vad_boundaries",
                "start_time": float(idx[0] / frame_hz),
                "end_time": float((idx[-1] + 1) / frame_hz),
                "num_frames": int(vals.size),
                "unit_nll": float(vals.mean()),
                "speaker": None,
                "partner": None,
                "detail": json.dumps({"reason": "no_valid_utterance_units"}, ensure_ascii=False),
            }]

    valid = scoring_valid_mask(len(nll), frame_hz, context_s)
    vals = nll[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []
    idx = np.flatnonzero(valid)
    return [{
        "segment_id": segment_id,
        "unit_id": 0,
        "unit_type": "fallback_all_valid",
        "start_time": float(idx[0] / frame_hz),
        "end_time": float((idx[-1] + 1) / frame_hz),
        "num_frames": int(vals.size),
        "unit_nll": float(vals.mean()),
        "speaker": None,
        "partner": None,
        "detail": json.dumps({"reason": "no_vad_boundaries"}, ensure_ascii=False),
    }]


def aggregate_unit_rows(unit_rows: list[dict[str, Any]], gamma: float, lam: float) -> dict[str, Any]:
    if not unit_rows:
        return {
            "mean_nll": float("nan"),
            "tail_nll": float("nan"),
            "dialog_nll": float("nan"),
            "combined_nll": float("nan"),
            "nat_score": float("nan"),
            "naturalness_score": float("nan"),
            "num_units": 0,
            "tail_k": 0,
            "unit_nll": [],
        }
    arr = np.asarray([float(row["unit_nll"]) for row in unit_rows], dtype=np.float64)
    tail_k = max(1, int(math.ceil(max(0.0, min(1.0, gamma)) * len(arr))))
    tail = np.sort(arr)[-tail_k:]
    mean_nll = float(arr.mean())
    tail_nll = float(tail.mean())
    dialog_nll = float(lam * mean_nll + (1.0 - lam) * tail_nll)
    return {
        "mean_nll": mean_nll,
        "tail_nll": tail_nll,
        "dialog_nll": dialog_nll,
        "combined_nll": dialog_nll,
        "nat_score": -dialog_nll,
        "naturalness_score": -dialog_nll,
        "num_units": int(len(unit_rows)),
        "tail_k": int(tail_k),
        "unit_nll": [float(x) for x in arr],
    }


@torch.no_grad()
def vap_frame_nll(
    model,
    audio: torch.Tensor,
    vad_50hz: np.ndarray,
    *,
    device: torch.device,
    chunk_s: float,
) -> dict[str, np.ndarray]:
    sr = int(model.sample_rate)
    frame_hz = float(model.frame_hz)
    chunk_samples = int(round(chunk_s * sr))
    if chunk_samples <= 0:
        chunk_samples = audio.shape[-1]

    frame_nll_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    n_samples = int(audio.shape[-1])
    for start in range(0, n_samples, chunk_samples):
        end = min(n_samples, start + chunk_samples)
        if end - start < sr:
            continue
        chunk = audio[:, start:end]
        frame_start = int(round((start / sr) * frame_hz))
        frame_end = int(round((end / sr) * frame_hz))
        vad_chunk_np = vad_50hz[frame_start:frame_end]
        if vad_chunk_np.size == 0:
            continue
        vad = torch.from_numpy(vad_chunk_np.astype(np.float32)).unsqueeze(0).to(device)
        out = model(chunk.unsqueeze(0).to(device))
        labels = model.objective.get_labels(vad).long()
        n = min(int(out["logits"].shape[1]), int(labels.shape[1]))
        if n <= 0:
            continue
        logits = out["logits"][:, :n]
        labels = labels[:, :n]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape(1, n)
        frame_nll_parts.append(loss.squeeze(0).detach().float().cpu().numpy())
        target_parts.append(labels.squeeze(0).detach().long().cpu().numpy())

    if not frame_nll_parts:
        return {"nll": np.zeros((0,), dtype=np.float32), "targets": np.zeros((0,), dtype=np.int64)}
    return {"nll": np.concatenate(frame_nll_parts).astype(np.float32), "targets": np.concatenate(target_parts).astype(np.int64)}


def frame_rows_for_segment(*, segment_id: str, condition: str, nll: np.ndarray, targets: np.ndarray, frame_hz: float, context_s: float) -> list[dict[str, Any]]:
    valid = scoring_valid_mask(len(nll), frame_hz, context_s)
    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(nll):
        rows.append({
            "segment_id": segment_id,
            "condition": condition,
            "frame_idx": int(idx),
            "time": float(idx / frame_hz),
            "target_256": int(targets[idx]) if idx < len(targets) else -1,
            "nll": float(value),
            "scoring_valid": bool(valid[idx]),
        })
    return rows


def score_audio(
    model,
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
    chunk_s: float,
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
    save_frame_scores: bool,
) -> dict[str, Any]:
    audio = load_stereo_audio(audio_path, int(model.sample_rate))
    sr = int(model.sample_rate)
    duration_s = float(audio.shape[-1] / sr)
    vad_pack = observed_vad_50hz(
        audio,
        sr=sr,
        vad_source=vad_source,
        silero_models=silero_models,
        rms_threshold=rms_threshold,
        silero_threshold=silero_threshold,
        silero_min_speech_ms=silero_min_speech_ms,
        silero_min_silence_ms=silero_min_silence_ms,
        clean_min_speech_ms=clean_min_speech_ms,
        clean_min_silence_ms=clean_min_silence_ms,
    )
    clean = vad_pack["clean"]
    frame_out = vap_frame_nll(model, audio, clean, device=device, chunk_s=chunk_s)
    nll = frame_out["nll"]
    targets = frame_out["targets"]
    units = extract_utterance_units(
        clean,
        frame_hz=float(model.frame_hz),
        duration_s=duration_s,
        unit_pre_s=unit_pre_s,
        unit_post_s=unit_post_s,
        min_utterance_s=min_utterance_s,
        unit_mode=unit_mode,
        utterance_merge_gap_s=utterance_merge_gap_s,
        utterance_merge_other_max_ratio=utterance_merge_other_max_ratio,
    )
    unit_rows = unit_rows_from_units(
        nll,
        units,
        vad=clean,
        segment_id=segment_id,
        frame_hz=float(model.frame_hz),
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
        "num_raw_vad_frames": int(vad_pack["raw"].shape[0]),
        "num_clean_vad_frames": int(clean.shape[0]),
        "unit_type_counts": dict(sorted(unit_counts.items())),
        "units": unit_rows,
        "frames": frame_rows_for_segment(segment_id=segment_id, condition=condition, nll=nll, targets=targets, frame_hz=float(model.frame_hz), context_s=context_s) if save_frame_scores else [],
    })
    return aggregate


def sample_stem_from_session(session_id: str) -> str:
    return session_id.split("__", 1)[0]


def natural_reference_from_meta(meta: dict[str, Any], json_path: str | Path) -> Path:
    natural_audio = Path(str(meta.get("natural_reference_wav", "")))
    if natural_audio.exists():
        return natural_audio
    raise FileNotFoundError(
        "Missing natural_reference_wav in generated JSON. This no-anchor scorer expects paired cropped natural references for pairwise diagnostics. "
        f"JSON={json_path}"
    )


def mean_var_ci95(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "var": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_half_width": float("nan"),
        }
    mean = float(arr.mean())
    var = float(arr.var(ddof=1)) if n > 1 else 0.0
    half = float(1.96 * math.sqrt(var) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean,
        "var": var,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
    }


def flatten_named_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def dialogue_nll_c_index(edited: np.ndarray, natural: np.ndarray) -> dict[str, float | int]:
    edited = np.asarray(edited, dtype=np.float64)
    natural = np.asarray(natural, dtype=np.float64)
    edited = edited[np.isfinite(edited)]
    natural = natural[np.isfinite(natural)]
    total = int(edited.size * natural.size)
    if total == 0:
        return {
            "dialogue_nll_c_index": float("nan"),
            "dialogue_nll_c_index_concordant": 0,
            "dialogue_nll_c_index_wrong": 0,
            "dialogue_nll_c_index_tied": 0,
            "dialogue_nll_c_index_total_pairs": 0,
        }
    diff = edited[:, None] - natural[None, :]
    concordant = int((diff > 0).sum())
    wrong = int((diff < 0).sum())
    tied = int((diff == 0).sum())
    comparable = concordant + wrong
    c_index = float(concordant / comparable) if comparable > 0 else float("nan")
    return {
        "dialogue_nll_c_index": c_index,
        "dialogue_nll_c_index_concordant": concordant,
        "dialogue_nll_c_index_wrong": wrong,
        "dialogue_nll_c_index_tied": tied,
        "dialogue_nll_c_index_total_pairs": total,
        "dialogue_nll_c_index_comparable_pairs": comparable,
    }


def nll_to_score_1_5(nll: float, nll_low: float, nll_high: float) -> float:
    if not math.isfinite(float(nll)) or not math.isfinite(nll_low) or not math.isfinite(nll_high):
        return float("nan")
    if nll_high <= nll_low:
        return 3.0
    score = 5.0 - 4.0 * ((float(nll) - nll_low) / (nll_high - nll_low))
    return float(np.clip(score, 1.0, 5.0))


def fit_score_1_5_scale(
    segment_rows: list[dict[str, Any]],
    *,
    low_quantile: float,
    high_quantile: float,
    fixed_low: float | None,
    fixed_high: float | None,
) -> dict[str, Any]:
    values = np.asarray([float(r["dialog_nll"]) for r in segment_rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        low = float("nan")
        high = float("nan")
        mode = "empty"
    elif fixed_low is not None and fixed_high is not None:
        low = float(fixed_low)
        high = float(fixed_high)
        mode = "fixed"
    else:
        q_low = float(np.clip(low_quantile, 0.0, 1.0))
        q_high = float(np.clip(high_quantile, 0.0, 1.0))
        if q_high <= q_low:
            q_low, q_high = 0.05, 0.95
        low = float(np.quantile(values, q_low))
        high = float(np.quantile(values, q_high))
        mode = "run_quantile"
    if math.isfinite(low) and math.isfinite(high) and high <= low:
        high = low + 1e-6
    return {
        "mode": mode,
        "nll_low_maps_to_5": low,
        "nll_high_maps_to_1": high,
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "num_segments": int(values.size),
        "formula": "score_1_5 = clip(5 - 4 * (DialogNLL - nll_low) / (nll_high - nll_low), 1, 5)",
    }


def add_score_1_5_fields(
    segment_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    scale: dict[str, Any],
) -> None:
    low = float(scale["nll_low_maps_to_5"])
    high = float(scale["nll_high_maps_to_1"])
    by_segment: dict[str, float] = {}
    for row in segment_rows:
        score = nll_to_score_1_5(float(row["dialog_nll"]), low, high)
        row["score_1_5"] = score
        row["score_1_5_nll_low"] = low
        row["score_1_5_nll_high"] = high
        by_segment[str(row["segment_id"])] = score
    for row in pair_rows:
        original = by_segment.get(str(row["original_segment_id"]), float("nan"))
        edited = by_segment.get(str(row["edited_segment_id"]), float("nan"))
        row["original_score_1_5"] = original
        row["edited_score_1_5"] = edited
        row["delta_score_1_5"] = original - edited if math.isfinite(original) and math.isfinite(edited) else float("nan")


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if math.isfinite(float(r["delta_nll"]))]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_type[str(row["edit_type"])].append(row)

    def pack(group: list[dict[str, Any]]) -> dict[str, Any]:
        deltas = np.asarray([float(r["delta_nll"]) for r in group], dtype=np.float64)
        edited = np.asarray([float(r["edited_dialog_nll"]) for r in group], dtype=np.float64)
        natural = np.asarray([float(r["original_dialog_nll"]) for r in group], dtype=np.float64)
        out = {
            "n": int(len(group)),
            "pairwise_accuracy": float((deltas > 0).mean()) if len(group) else float("nan"),
        }
        out.update(flatten_named_stats("delta_nll", mean_var_ci95(deltas)))
        out.update(flatten_named_stats("edited_nll", mean_var_ci95(edited)))
        out.update(flatten_named_stats("natural_nll", mean_var_ci95(natural)))
        out.update(dialogue_nll_c_index(edited, natural))
        if group and "delta_score_1_5" in group[0]:
            delta_scores = np.asarray([float(r.get("delta_score_1_5", float("nan"))) for r in group], dtype=np.float64)
            edited_scores = np.asarray([float(r.get("edited_score_1_5", float("nan"))) for r in group], dtype=np.float64)
            natural_scores = np.asarray([float(r.get("original_score_1_5", float("nan"))) for r in group], dtype=np.float64)
            out.update(flatten_named_stats("delta_score_1_5", mean_var_ci95(delta_scores)))
            out.update(flatten_named_stats("edited_score_1_5", mean_var_ci95(edited_scores)))
            out.update(flatten_named_stats("natural_score_1_5", mean_var_ci95(natural_scores)))
        return out

    return {"overall": pack(ok_rows), "by_type": {k: pack(v) for k, v in sorted(by_type.items())}}




def summarize_protected_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy helper kept for old result files; active scorer no longer calls this."""
    ok_rows = [r for r in rows if math.isfinite(float(r.get("protected_delta_nll", float("nan"))))]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_type[str(row["edit_type"])].append(row)

    def pack(group: list[dict[str, Any]]) -> dict[str, Any]:
        deltas = np.asarray([float(r["protected_delta_nll"]) for r in group], dtype=np.float64)
        edited = np.asarray([float(r["edited_protected_dialog_nll"]) for r in group], dtype=np.float64)
        natural = np.asarray([float(r["original_protected_dialog_nll"]) for r in group], dtype=np.float64)
        out = {
            "n": int(len(group)),
            "pairwise_accuracy": float((deltas > 0).mean()) if len(group) else float("nan"),
        }
        out.update(flatten_named_stats("delta_nll", mean_var_ci95(deltas)))
        out.update(flatten_named_stats("edited_nll", mean_var_ci95(edited)))
        out.update(flatten_named_stats("natural_nll", mean_var_ci95(natural)))
        out.update(dialogue_nll_c_index(edited, natural))
        return out

    return {"overall": pack(ok_rows), "by_type": {k: pack(v) for k, v in sorted(by_type.items())}}


def summarize_segments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in rows if math.isfinite(float(r["dialog_nll"]))]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        condition = str(row["condition"])
        edit_type = str(row["edit_type"])
        by_condition[condition].append(row)
        by_type_condition[f"{edit_type}/{condition}"].append(row)

    def stats(values: list[float]) -> dict[str, float]:
        return mean_var_ci95(values)

    def pack(group: list[dict[str, Any]]) -> dict[str, Any]:
        out = {
            "n": int(len(group)),
            "mean_nll": stats([float(r["mean_nll"]) for r in group]),
            "tail_nll": stats([float(r["tail_nll"]) for r in group]),
            "dialog_nll": stats([float(r["dialog_nll"]) for r in group]),
            "nat_score": stats([float(r["nat_score"]) for r in group]),
        }
        if group and "score_1_5" in group[0]:
            out["score_1_5"] = stats([float(r.get("score_1_5", float("nan"))) for r in group])
        return out

    return {
        "overall": pack(ok_rows),
        "by_condition": {k: pack(v) for k, v in sorted(by_condition.items())},
        "by_type_condition": {k: pack(v) for k, v in sorted(by_type_condition.items())},
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="No-anchor VAP 256-state utterance-boundary NLL naturalness scorer.")
    ap.add_argument("--unnatural-manifest", type=Path, default=DEFAULT_UNNATURAL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vad-source", choices=["silero", "rms"], default="silero")
    ap.add_argument("--rms-threshold", type=float, default=0.015)
    ap.add_argument("--silero-threshold", type=float, default=0.5)
    ap.add_argument("--silero-min-speech-ms", type=int, default=100)
    ap.add_argument("--silero-min-silence-ms", type=int, default=50)
    ap.add_argument("--clean-min-speech-ms", type=int, default=150)
    ap.add_argument("--clean-min-silence-ms", type=int, default=150)
    ap.add_argument("--chunk-s", type=float, default=60.0)
    ap.add_argument("--context-s", type=float, default=3.0)
    ap.add_argument("--tail-gamma", type=float, default=0.25)
    ap.add_argument("--lambda-mean", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None, help="Deprecated alias: tail_gamma = 1 - tau.")
    ap.add_argument("--min-unit-frames", type=int, default=5)
    ap.add_argument("--unit-pre-s", type=float, default=2.0)
    ap.add_argument("--unit-post-s", type=float, default=0.0, help="Seconds after each utterance boundary included in boundary units. Default 0 scores only pre-boundary frames.")
    ap.add_argument("--min-utterance-s", type=float, default=0.5)
    ap.add_argument("--utterance-merge-gap-s", type=float, default=1.0)
    ap.add_argument("--utterance-merge-other-max-ratio", type=float, default=0.2)
    ap.add_argument("--unit-mode", choices=["boundaries", "spans", "both"], default="boundaries")
    ap.add_argument("--save-frame-scores", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--score-low-quantile", type=float, default=0.05, help="Run-level DialogNLL quantile mapped to score 5. Ignored if fixed score anchors are provided.")
    ap.add_argument("--score-high-quantile", type=float, default=0.95, help="Run-level DialogNLL quantile mapped to score 1. Ignored if fixed score anchors are provided.")
    ap.add_argument("--score-nll-low", type=float, default=None, help="Fixed DialogNLL anchor that maps to score 5 for cross-run comparable scoring.")
    ap.add_argument("--score-nll-high", type=float, default=None, help="Fixed DialogNLL anchor that maps to score 1 for cross-run comparable scoring.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.tau is not None:
        args.tail_gamma = max(0.0, min(1.0, 1.0 - args.tau))

    device = torch.device(args.device)
    model = load_vap_model(args.checkpoint, device)
    silero_models = load_silero_models() if args.vad_source == "silero" else None
    rows = read_manifest(args.unnatural_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    out_dir = args.output_dir / "artifacts" / "vap_nll_naturalness"
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    unit_rows_all: list[dict[str, Any]] = []
    frame_rows_all: list[dict[str, Any]] = []

    row_iter = tqdm(
        rows,
        total=len(rows),
        desc="VAP NLL pairs",
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
            })
            unit_rows_all.extend(scored["units"])
            frame_rows_all.extend(scored["frames"])

        delta_nll = float(edited["dialog_nll"]) - float(natural["dialog_nll"])
        delta_score = float(natural["nat_score"]) - float(edited["nat_score"])
        pair = {
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
        }
        pair_rows.append(pair)
        print(
            f"[{i:3d}/{len(rows)}] {session_id} type={edit_type} "
            f"orig_nll={natural['dialog_nll']:.4f} edited_nll={edited['dialog_nll']:.4f} "
            f"delta={delta_nll:.4f} "
            f"edited>{'yes ' if delta_nll > 0 else 'no '}"
            f"units={natural['num_units']}/{edited['num_units']}"
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
    legacy_csv = out_dir / "vap_nll_scores.csv"

    pair_fields = list(pair_rows[0].keys()) if pair_rows else []
    segment_fields = list(segment_rows[0].keys()) if segment_rows else []
    unit_fields = list(unit_rows_all[0].keys()) if unit_rows_all else ["segment_id", "unit_id", "unit_type", "start_time", "end_time", "num_frames", "unit_nll", "speaker", "partner", "detail"]
    frame_fields = list(frame_rows_all[0].keys()) if frame_rows_all else ["segment_id", "condition", "frame_idx", "time", "target_256", "nll", "scoring_valid"]
    write_csv(pair_csv, pair_rows, pair_fields)
    write_csv(legacy_csv, pair_rows, pair_fields)
    write_csv(segment_csv, segment_rows, segment_fields)
    write_csv(unit_csv, unit_rows_all, unit_fields)
    if args.save_frame_scores:
        write_csv(frame_csv, frame_rows_all, frame_fields)

    payload = {
        "metric": "vap_256_state_utterance_boundary_nll_no_anchor",
        "config": {
            "unnatural_manifest": str(args.unnatural_manifest),
            "checkpoint": str(args.checkpoint),
            "vad_source": args.vad_source,
            "rms_threshold": args.rms_threshold,
            "silero_threshold": args.silero_threshold,
            "silero_min_speech_ms": args.silero_min_speech_ms,
            "silero_min_silence_ms": args.silero_min_silence_ms,
            "clean_min_speech_ms": args.clean_min_speech_ms,
            "clean_min_silence_ms": args.clean_min_silence_ms,
            "chunk_s": args.chunk_s,
            "context_s": args.context_s,
            "tail_gamma": args.tail_gamma,
            "lambda_mean": args.lambda_mean,
            "mean_nll_coefficient": args.lambda_mean,
            "tail_nll_coefficient": 1.0 - args.lambda_mean,
            "min_unit_frames": args.min_unit_frames,
            "unit_definition": "VAD utterance boundary windows after filtering very short speech islands and merging same-speaker islands separated by short pauses with little other-speaker activity; each merged utterance contributes start/end units. Default boundary window is [boundary - unit_pre_s, boundary + unit_post_s], with unit_post_s=0 for pre-boundary scoring.",
            "unit_mode": args.unit_mode,
            "unit_pre_s": args.unit_pre_s,
            "unit_post_s": args.unit_post_s,
            "min_utterance_s": args.min_utterance_s,
            "utterance_merge_gap_s": args.utterance_merge_gap_s,
            "utterance_merge_other_max_ratio": args.utterance_merge_other_max_ratio,
            "score_1_5_scale": score_scale,
            "global_unit_definition": "VAD utterance units over the whole clip; scorer ignores edit metadata and uses one unified no-anchor utterance-boundary metric.",
            "no_anchor_guarantee": "Scoring ignores edit metadata and uses no protected/event-specific windows.",
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
    json_path = out_dir / "vap_nll_results.json"
    write_json(json_path, payload)

    print("=" * 90)
    print("No-anchor VAP utterance-boundary NLL summary")
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
