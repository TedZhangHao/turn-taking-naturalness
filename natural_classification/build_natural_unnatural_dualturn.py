#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is a convenience dependency.
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_postfix(self, *args, **kwargs):
            return None

    def tqdm(iterable=None, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

SILERO_CONFIG: dict[str, float] = {
    "threshold": 0.50,
    "min_speech_ms": 120.0,
    "min_silence_ms": 120.0,
    "speech_pad_ms": 40.0,
    "merge_gap_ms": 180.0,
}
_SILERO_MODEL: Any | None = None
_SILERO_SEGMENT_CACHE: dict[tuple[Any, ...], tuple[list["Segment"], str]] = {}

MANIFEST_FIELDS = [
    "id",
    "session_id",
    "source_type",
    "audio_path",
    "json_path",
    "tar_path",
    "member_flac",
    "member_json",
    "duration_sec",
    "language",
    "session_type",
    "split",
]

UNNATURAL_TYPES = [
    "late_response",
    "early_entry",
    "missed_backchannel",
    "excessive_backchannel",
    "hold_instead_of_shift",
    "shift_instead_of_hold",
]


@dataclass
class Segment:
    start: int
    end: int
    text: str = ""
    source: str = "rms"
    words: list[dict[str, Any]] | None = None
    start_s: float | None = None
    end_s: float | None = None

    @property
    def dur(self) -> int:
        return self.end - self.start + 1


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compact_failure_reason(reason: str, max_len: int = 96) -> str:
    text = str(reason).replace("\n", " ").strip()
    if "; shift_hold_diagnostics=" in text:
        text = text.split("; shift_hold_diagnostics=", 1)[0]
    if "; diagnostics=" in text:
        text = text.split("; diagnostics=", 1)[0]
    return text[:max_len]


def is_deterministic_no_candidate_failure(edit_type: str, reason: str) -> bool:
    """Avoid retrying deterministic candidate searches on the same source."""
    text = str(reason)
    if edit_type == "hold_instead_of_shift" and "No textual non-question shift-return found" in text:
        return True
    return False


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_mono(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    return x.astype(np.float32), sr


def read_stereo(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True)
    x = x.astype(np.float32)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    return x, sr


def pad_to_same_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    return a, b


def rms_vad(x: np.ndarray, sr: int, frame_ms: float = 80.0, threshold: float = 0.015) -> np.ndarray:
    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    pad = (-len(x)) % frame_len
    if pad:
        x = np.pad(x, (0, pad))
    frames = x.reshape(-1, frame_len)
    rms = np.sqrt(np.maximum((frames ** 2).mean(axis=1), 1e-8))
    return (rms >= threshold).astype(np.uint8)


def segments_from_vad(vad: np.ndarray) -> list[Segment]:
    segs: list[Segment] = []
    active = False
    start = 0
    for i, v in enumerate(vad):
        if v and not active:
            start = i
            active = True
        elif not v and active:
            segs.append(Segment(start, i - 1))
            active = False
    if active:
        segs.append(Segment(start, len(vad) - 1))
    return segs




def _sidecar_json_path(audio_path: str | Path | None) -> Path | None:
    if not audio_path:
        return None
    path = Path(audio_path)
    cand = path.with_suffix(".json")
    return cand if cand.exists() else None


def _metadata_list(meta: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        val = meta.get(key)
        if isinstance(val, list):
            return val
    nested = meta.get("metadata")
    if isinstance(nested, dict):
        for key in keys:
            short = key.split(":", 1)[-1]
            val = nested.get(short)
            if isinstance(val, list):
                return val
    return []


def _load_transcript_items(audio_path: str | Path | None) -> list[dict[str, Any]]:
    sidecar = _sidecar_json_path(audio_path)
    if sidecar is None:
        return []
    meta = _read_json(sidecar)
    return _metadata_list(meta, "metadata:transcript", "transcript")


def _sec_to_segment(
    start_s: float,
    end_s: float,
    frame_ms: float,
    text: str,
    source: str,
    words: list[dict[str, Any]] | None = None,
) -> Segment | None:
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        return None
    start = max(0, int(math.floor(start_s * 1000.0 / frame_ms)))
    end = max(start, int(math.ceil(end_s * 1000.0 / frame_ms)) - 1)
    return Segment(start=start, end=end, text=text, source=source, words=words, start_s=start_s, end_s=end_s)


def _overlapping_transcript(
    transcript_items: list[dict[str, Any]],
    start_s: float,
    end_s: float,
) -> tuple[str, list[dict[str, Any]] | None, float | None, float | None]:
    texts: list[str] = []
    words: list[dict[str, Any]] = []
    bounds: list[tuple[float, float]] = []
    for tr in transcript_items:
        try:
            tr_start = float(tr["start"])
            tr_end = float(tr["end"])
        except Exception:
            continue
        if min(end_s, tr_end) <= max(start_s, tr_start):
            continue
        bounds.append((tr_start, tr_end))
        text = str(tr.get("transcript") or "").strip()
        tr_words = tr.get("words")
        word_texts: list[str] = []
        if isinstance(tr_words, list):
            for word in tr_words:
                if not isinstance(word, dict):
                    continue
                try:
                    word_start = float(word.get("start", tr_start))
                    word_end = float(word.get("end", tr_end))
                except Exception:
                    words.append(word)
                    token = str(word.get("word") or word.get("text") or "").strip()
                    if token:
                        word_texts.append(token)
                    continue
                if min(end_s, word_end) > max(start_s, word_start):
                    words.append(word)
                    token = str(word.get("word") or word.get("text") or "").strip()
                    if token:
                        word_texts.append(token)
        if word_texts:
            texts.append(" ".join(word_texts))
        elif text:
            texts.append(text)
    text = " ".join(texts)
    if words:
        word_bounds = []
        for word in words:
            try:
                word_bounds.append((float(word["start"]), float(word["end"])))
            except Exception:
                pass
        if word_bounds:
            bounds.extend(word_bounds)
    if not bounds:
        return text, (words or None), None, None
    return text, (words or None), min(x[0] for x in bounds), max(x[1] for x in bounds)


def _resample_mono_float32(x: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr == target_sr:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if len(x) == 0:
        return x
    try:
        from scipy.signal import resample_poly

        g = math.gcd(int(sr), int(target_sr))
        y = resample_poly(x, target_sr // g, sr // g)
    except Exception:
        old_t = np.arange(len(x), dtype=np.float64) / float(sr)
        new_n = max(1, int(round(len(x) * float(target_sr) / float(sr))))
        new_t = np.arange(new_n, dtype=np.float64) / float(target_sr)
        y = np.interp(new_t, old_t, x).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _get_silero_model() -> Any:
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        try:
            from silero_vad import load_silero_vad
        except Exception as e:
            raise RuntimeError(
                "silero_vad is required for --turn-source silero. "
                "Install it with: python -m pip install silero-vad"
            ) from e
        _SILERO_MODEL = load_silero_vad()
    return _SILERO_MODEL


def _clone_segments(segs: list[Segment]) -> list[Segment]:
    return [
        Segment(
            start=s.start,
            end=s.end,
            text=s.text,
            source=s.source,
            words=list(s.words) if s.words else None,
            start_s=s.start_s,
            end_s=s.end_s,
        )
        for s in segs
    ]


def _merge_close_segments(
    segs: list[Segment],
    *,
    frame_ms: float,
    merge_gap_ms: float,
) -> list[Segment]:
    if not segs or merge_gap_ms <= 0:
        return sorted(segs, key=lambda s: (s.start, s.end))
    max_gap_frames = int(math.ceil(merge_gap_ms / frame_ms))
    out: list[Segment] = []
    for seg in sorted(segs, key=lambda s: (s.start, s.end)):
        if not out or seg.start - out[-1].end - 1 > max_gap_frames:
            out.append(seg)
            continue
        prev = out[-1]
        prev.end = max(prev.end, seg.end)
        if prev.start_s is None or (seg.start_s is not None and seg.start_s < prev.start_s):
            prev.start_s = seg.start_s
        if prev.end_s is None or (seg.end_s is not None and seg.end_s > prev.end_s):
            prev.end_s = seg.end_s
        old_text = (prev.text or "").strip()
        new_text = (seg.text or "").strip()
        old_norm = re.sub(r"\s+", " ", old_text.lower()).strip()
        new_norm = re.sub(r"\s+", " ", new_text.lower()).strip()
        if not old_text:
            prev.text = new_text
        elif new_text and old_norm not in new_norm and new_norm not in old_norm:
            prev.text = f"{old_text} {new_text}".strip()
        elif new_text and len(new_text) > len(old_text):
            prev.text = new_text
        if prev.words or seg.words:
            seen = set()
            merged_words = []
            for word in list(prev.words or []) + list(seg.words or []):
                key = (word.get("word"), word.get("start"), word.get("end")) if isinstance(word, dict) else id(word)
                if key in seen:
                    continue
                seen.add(key)
                merged_words.append(word)
            prev.words = merged_words
    return out


def load_silero_segments(
    x: np.ndarray,
    sr: int,
    audio_path: str | Path | None,
    *,
    frame_ms: float,
    min_transcript_ms: float = 40.0,
) -> tuple[list[Segment], str]:
    try:
        import torch
        from silero_vad import get_speech_timestamps
    except Exception as e:
        raise RuntimeError(
            "silero_vad + torch are required for --turn-source silero. "
            "Install it with: python -m pip install silero-vad"
        ) from e

    cache_key = None
    if audio_path is not None:
        cache_key = (
            str(Path(audio_path).resolve()),
            int(sr),
            float(frame_ms),
            float(min_transcript_ms),
            tuple(sorted((k, float(v)) for k, v in SILERO_CONFIG.items())),
        )
        cached = _SILERO_SEGMENT_CACHE.get(cache_key)
        if cached is not None:
            segs, source = cached
            return _clone_segments(segs), source

    target_sr = 16000
    wav = _resample_mono_float32(x, sr, target_sr)
    if wav.size == 0:
        return [], "silero_vad_empty"

    model = _get_silero_model()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    ts = get_speech_timestamps(
        torch.from_numpy(wav),
        model,
        sampling_rate=target_sr,
        threshold=float(SILERO_CONFIG["threshold"]),
        min_speech_duration_ms=int(round(SILERO_CONFIG["min_speech_ms"])),
        min_silence_duration_ms=int(round(SILERO_CONFIG["min_silence_ms"])),
        speech_pad_ms=int(round(SILERO_CONFIG["speech_pad_ms"])),
        return_seconds=True,
    )

    transcript_items = _load_transcript_items(audio_path)
    out: list[Segment] = []
    for item in ts:
        try:
            start_s = float(item["start"])
            end_s = float(item["end"])
        except Exception:
            continue
        if (end_s - start_s) * 1000.0 < float(min_transcript_ms):
            continue
        text, words, _, _ = _overlapping_transcript(transcript_items, start_s, end_s)
        seg = _sec_to_segment(start_s, end_s, frame_ms, text, "silero_vad", words=words)
        if seg is not None:
            out.append(seg)

    out = _merge_close_segments(out, frame_ms=frame_ms, merge_gap_ms=float(SILERO_CONFIG["merge_gap_ms"]))
    cfg = (
        "silero_vad"
        f"(threshold={SILERO_CONFIG['threshold']},"
        f"min_speech_ms={SILERO_CONFIG['min_speech_ms']},"
        f"min_silence_ms={SILERO_CONFIG['min_silence_ms']},"
        f"speech_pad_ms={SILERO_CONFIG['speech_pad_ms']},"
        f"merge_gap_ms={SILERO_CONFIG['merge_gap_ms']})"
    )
    if cache_key is not None:
        _SILERO_SEGMENT_CACHE[cache_key] = (_clone_segments(out), cfg)
    return out, cfg


def load_sidecar_segments(
    audio_path: str | Path | None,
    *,
    frame_ms: float,
    min_transcript_ms: float = 250.0,
    prefer_transcript: bool = True,
) -> tuple[list[Segment], str]:
    """
    Prefer naturalistic sidecar timestamps over audio-energy VAD.

    The original seamless/naturalistic files usually have a sibling JSON with
    metadata:transcript and metadata:vad. Transcript utterances are less likely
    to treat small room noise as a conversational turn, so they are the primary
    source for edit candidates. If transcript is unavailable, fall back to the
    provided metadata VAD; if both are missing, the caller can still use RMS VAD.
    """
    sidecar = _sidecar_json_path(audio_path)
    if sidecar is None:
        return [], "missing_sidecar"
    meta = _read_json(sidecar)
    transcript_items = _metadata_list(meta, "metadata:transcript", "transcript")

    if prefer_transcript:
        out: list[Segment] = []
        for item in transcript_items:
            try:
                start_s = float(item["start"])
                end_s = float(item["end"])
            except Exception:
                continue
            dur_ms = (end_s - start_s) * 1000.0
            text = str(item.get("transcript") or "").strip()
            if dur_ms < min_transcript_ms:
                continue
            words = item.get("words") if isinstance(item.get("words"), list) else None
            seg = _sec_to_segment(start_s, end_s, frame_ms, text, "transcript", words=words)
            if seg is not None:
                out.append(seg)
        if out:
            return sorted(out, key=lambda s: (s.start, s.end)), "transcript"

    out = []
    for item in _metadata_list(meta, "metadata:vad", "vad"):
        try:
            start_s = float(item["start"])
            end_s = float(item["end"])
        except Exception:
            continue
        text, words, text_start_s, text_end_s = _overlapping_transcript(transcript_items, start_s, end_s)
        # Use VAD for turn structure, but do not let a coarse VAD boundary cut
        # through the first/last transcript word when synthesizing edits.
        seg_start_s = start_s
        seg_end_s = end_s
        if text_start_s is not None and 0.0 <= start_s - text_start_s <= 0.5:
            seg_start_s = text_start_s
        if text_end_s is not None and 0.0 <= text_end_s - end_s <= 0.5:
            seg_end_s = text_end_s
        seg = _sec_to_segment(seg_start_s, seg_end_s, frame_ms, text, "metadata_vad", words=words)
        if seg is not None:
            out.append(seg)
    if out:
        return sorted(out, key=lambda s: (s.start, s.end)), "metadata_vad"

    return [], "empty_sidecar"


def segments_to_vad(segs: list[Segment], n_frames: int) -> np.ndarray:
    vad = np.zeros((max(0, n_frames),), dtype=np.uint8)
    for seg in segs:
        s = max(0, min(n_frames, seg.start))
        e = max(0, min(n_frames, seg.end + 1))
        if e > s:
            vad[s:e] = 1
    return vad


def get_candidate_context(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    frame_ms: float,
    threshold: float,
    *,
    p1_path: str | Path | None = None,
    p2_path: str | Path | None = None,
    turn_source: str = "silero",
    min_transcript_ms: float = 250.0,
) -> tuple[list[Segment], list[Segment], np.ndarray, np.ndarray, dict[str, str]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    n_frames = int(math.ceil(max(len(a), len(b)) / float(frame_len)))
    sources = {"a": "rms", "b": "rms"}

    if turn_source == "rms":
        va, vb = rms_vad(a, sr, frame_ms, threshold), rms_vad(b, sr, frame_ms, threshold)
        return segments_from_vad(va), segments_from_vad(vb), va, vb, sources

    if turn_source == "silero":
        sa, source_a = load_silero_segments(
            a, sr, p1_path, frame_ms=frame_ms, min_transcript_ms=min_transcript_ms
        )
        sb, source_b = load_silero_segments(
            b, sr, p2_path, frame_ms=frame_ms, min_transcript_ms=min_transcript_ms
        )
        sources = {"a": source_a, "b": source_b}

        if not sa:
            va = rms_vad(a, sr, frame_ms, threshold)
            sa = segments_from_vad(va)
            sources["a"] = f"{source_a}->rms"
        else:
            va = segments_to_vad(sa, n_frames)

        if not sb:
            vb = rms_vad(b, sr, frame_ms, threshold)
            sb = segments_from_vad(vb)
            sources["b"] = f"{source_b}->rms"
        else:
            vb = segments_to_vad(sb, n_frames)

        return sa, sb, va, vb, sources

    prefer_transcript = turn_source != "metadata_vad"
    sa, source_a = load_sidecar_segments(
        p1_path, frame_ms=frame_ms, min_transcript_ms=min_transcript_ms, prefer_transcript=prefer_transcript
    )
    sb, source_b = load_sidecar_segments(
        p2_path, frame_ms=frame_ms, min_transcript_ms=min_transcript_ms, prefer_transcript=prefer_transcript
    )
    sources = {"a": source_a, "b": source_b}

    if not sa:
        va = rms_vad(a, sr, frame_ms, threshold)
        sa = segments_from_vad(va)
        sources["a"] = f"{source_a}->rms"
    else:
        va = segments_to_vad(sa, n_frames)

    if not sb:
        vb = rms_vad(b, sr, frame_ms, threshold)
        sb = segments_from_vad(vb)
        sources["b"] = f"{source_b}->rms"
    else:
        vb = segments_to_vad(sb, n_frames)

    return sa, sb, va, vb, sources

def frame_to_sample(idx: int, frame_len: int) -> int:
    return idx * frame_len


def frame_to_ms(idx: int, frame_ms: float) -> float:
    return round(float(idx) * float(frame_ms), 1)


def sample_to_ms(idx: int, sr: int) -> float:
    return round(float(idx) * 1000.0 / float(sr), 1)


def segment_start_sample(seg: Segment, sr: int, frame_len: int) -> int:
    if seg.start_s is not None:
        return max(0, int(round(seg.start_s * sr)))
    return frame_to_sample(seg.start, frame_len)


def segment_end_sample(seg: Segment, sr: int, frame_len: int) -> int:
    if seg.end_s is not None:
        return max(0, int(round(seg.end_s * sr)))
    return frame_to_sample(seg.end + 1, frame_len)


def segment_start_ms(seg: Segment, frame_ms: float) -> float:
    if seg.start_s is not None:
        return round(seg.start_s * 1000.0, 1)
    return frame_to_ms(int(seg.start), frame_ms)


def segment_end_ms(seg: Segment, frame_ms: float) -> float:
    if seg.end_s is not None:
        return round(seg.end_s * 1000.0, 1)
    return frame_to_ms(int(seg.end + 1), frame_ms)


def segment_duration_ms(seg: Segment, frame_ms: float) -> float:
    return round(segment_end_ms(seg, frame_ms) - segment_start_ms(seg, frame_ms), 1)




def segment_text_between_ms(seg: Segment, start_ms: float, end_ms: float) -> str:
    if not seg.words:
        return seg.text
    start_s = start_ms / 1000.0
    end_s = end_ms / 1000.0
    words: list[str] = []
    for item in seg.words:
        try:
            ws = float(item["start"])
            we = float(item["end"])
        except Exception:
            continue
        mid = (ws + we) / 2.0
        if start_s <= mid <= end_s:
            words.append(str(item.get("word") or "").strip())
    return " ".join(w for w in words if w).strip() or seg.text



def _word_text(word: dict[str, Any]) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _word_bounds_s(word: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start_s = float(word["start"])
        end_s = float(word["end"])
    except Exception:
        return None
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        return None
    return start_s, end_s


def _is_backchannel_text(text: str) -> bool:
    norm = normalized_text(text)
    if not norm:
        return False
    if norm in BACKCHANNEL_PHRASES:
        return True
    starts = ("yeah", "yep", "yes", "right", "ok", "okay", "mm", "mhm", "hm", "uh", "um", "oh")
    return norm.startswith(starts) and segment_word_count(Segment(0, 0, text=text)) <= 4


def backchannel_clip_bounds(
    sig: np.ndarray,
    seg: Segment,
    *,
    sr: int,
    frame_len: int,
    max_words: int = 4,
    max_duration_ms: float = 1200.0,
    word_pad_ms: float = 30.0,
) -> tuple[int, int, str, str]:
    """Return sample bounds for the audible backchannel, preferring word timestamps.

    Silero segments are often wider than the lexical backchannel. For
    excessive-backchannel edits we copy audio, so using word boundaries keeps
    metadata text aligned with the inserted sound.
    """
    fallback_start = segment_start_sample(seg, sr, frame_len)
    fallback_end = segment_end_sample(seg, sr, frame_len)
    fallback_text = (seg.text or "").strip()

    if not seg.words:
        return fallback_start, fallback_end, fallback_text, "vad_segment"

    seg_start_s = seg.start_s if seg.start_s is not None else fallback_start / float(sr)
    seg_end_s = seg.end_s if seg.end_s is not None else fallback_end / float(sr)
    words: list[tuple[int, float, float, str]] = []
    for idx, word in enumerate(seg.words):
        if not isinstance(word, dict):
            continue
        bounds = _word_bounds_s(word)
        text = _word_text(word)
        if bounds is None or not text:
            continue
        start_s, end_s = bounds
        if min(seg_end_s, end_s) <= max(seg_start_s, start_s):
            continue
        words.append((idx, start_s, end_s, text))

    spans: list[tuple[int, float, int, float, float, str]] = []
    for i in range(len(words)):
        for j in range(i, min(len(words), i + max_words)):
            start_s = words[i][1]
            end_s = words[j][2]
            duration_ms = (end_s - start_s) * 1000.0
            if duration_ms <= 0 or duration_ms > max_duration_ms:
                continue
            text = " ".join(w[3] for w in words[i:j + 1]).strip()
            if _is_backchannel_text(text):
                exact = 0 if normalized_text(text) in BACKCHANNEL_PHRASES else 1
                spans.append((exact, duration_ms, j - i + 1, start_s, end_s, text))

    if not spans:
        return fallback_start, fallback_end, fallback_text, "vad_segment_no_word_match"

    _, _, _, start_s, end_s, text = min(spans, key=lambda x: (x[0], x[1], x[2], x[3]))
    pad_s = max(0.0, float(word_pad_ms)) / 1000.0
    start = max(0, int(round((start_s - pad_s) * sr)))
    end = min(len(sig), int(round((end_s + pad_s) * sr)))
    if end <= start:
        return fallback_start, fallback_end, fallback_text, "vad_segment_empty_word_clip"
    return start, end, text, "word_timestamps"


def copy_clip(sig: np.ndarray, s: int, e: int) -> np.ndarray:
    s = max(0, s)
    e = min(len(sig), e)
    if e <= s:
        return np.zeros((0,), dtype=np.float32)
    return sig[s:e].copy()


def zero_region(sig: np.ndarray, s: int, e: int) -> None:
    s = max(0, s)
    e = min(len(sig), e)
    if e > s:
        sig[s:e] = 0.0


def paste_add(sig: np.ndarray, clip: np.ndarray, start: int, gain: float = 1.0) -> None:
    if clip.size == 0:
        return
    start = max(0, start)
    end = min(len(sig), start + len(clip))
    if end <= start:
        return
    n = end - start
    sig[start:end] += gain * clip[:n]
    np.clip(sig, -1.0, 1.0, out=sig)


def paste_replace(sig: np.ndarray, clip: np.ndarray, start: int, gain: float = 1.0) -> None:
    if clip.size == 0:
        return
    start = max(0, start)
    end = min(len(sig), start + len(clip))
    if end <= start:
        return
    n = end - start
    sig[start:end] = gain * clip[:n]
    np.clip(sig, -1.0, 1.0, out=sig)




def _fade_curve(n: int, reverse: bool = False) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    curve = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, n, dtype=np.float32))
    return curve[::-1] if reverse else curve


def apply_edge_fades(sig: np.ndarray, start: int, end: int, sr: int, fade_ms: float = 12.0) -> None:
    start = max(0, min(len(sig), start))
    end = max(start, min(len(sig), end))
    if end <= start:
        return
    fade = min((end - start) // 2, max(1, int(round(sr * fade_ms / 1000.0))))
    if fade <= 1:
        return
    sig[start:start + fade] *= _fade_curve(fade)
    sig[end - fade:end] *= _fade_curve(fade, reverse=True)


def zero_region_smooth(sig: np.ndarray, s: int, e: int, sr: int, fade_ms: float = 12.0) -> None:
    s = max(0, min(len(sig), s))
    e = max(s, min(len(sig), e))
    if e <= s:
        return
    fade = min((e - s) // 2, max(1, int(round(sr * fade_ms / 1000.0))))
    if fade > 1:
        left = sig[s:s + fade].copy() * _fade_curve(fade, reverse=True)
        right = sig[e - fade:e].copy() * _fade_curve(fade)
        sig[s:e] = 0.0
        sig[s:s + fade] = left
        sig[e - fade:e] = right
    else:
        sig[s:e] = 0.0


def replace_region_with_fill_smooth(
    sig: np.ndarray,
    s: int,
    e: int,
    fill: np.ndarray | None,
    sr: int,
    fade_ms: float = 12.0,
) -> None:
    s = max(0, min(len(sig), int(s)))
    e = max(s, min(len(sig), int(e)))
    if e <= s:
        return
    n = e - s
    fill_full = _fit_fill_channel(fill, n).astype(np.float32, copy=True)
    fade = min(n // 2, max(1, int(round(sr * fade_ms / 1000.0))))
    if fade > 1:
        fill_full[:fade] *= _fade_curve(fade)
        fill_full[n - fade:n] *= _fade_curve(fade, reverse=True)
    sig[s:e] = fill_full[:n]
    np.clip(sig, -1.0, 1.0, out=sig)


def shift_pair_earlier_smooth(
    a: np.ndarray,
    b: np.ndarray,
    cut_start: int,
    cut_len: int,
    sr: int,
    fade_ms: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    a2, b2 = shift_pair_earlier(a, b, cut_start, cut_len, sr=sr)
    fade = max(1, int(round(sr * fade_ms / 1000.0)))
    pos = max(0, min(len(a2), cut_start))
    lo = max(0, pos - fade)
    hi = min(len(a2), pos + fade)
    apply_edge_fades(a2, lo, hi, sr, fade_ms=fade_ms)
    apply_edge_fades(b2, lo, hi, sr, fade_ms=fade_ms)
    return a2, b2


def _has_text(seg: Segment, min_words: int = 2) -> bool:
    return segment_word_count(seg) >= min_words


def _distinct_text(a: Segment, b: Segment) -> bool:
    ta = normalized_text(a.text or "")
    tb = normalized_text(b.text or "")
    return bool(ta and tb and ta != tb)


def _is_question_like(seg: Segment) -> bool:
    text = str(seg.text or "")
    return "?" in text or "\uff1f" in text or "\u00bf" in text


def shift_pair_later_channel_only(
    a: np.ndarray,
    b: np.ndarray,
    channel: str,
    start: int,
    delay: int,
    fill: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if delay <= 0:
        return a, b
    start = max(0, min(len(a), start))
    fa, fb = fill
    if channel == "a":
        a = np.concatenate([a[:start], fa[:delay], a[start:]]).astype(np.float32)
        b = np.pad(b, (0, delay)).astype(np.float32)
    else:
        b = np.concatenate([b[:start], fb[:delay], b[start:]]).astype(np.float32)
        a = np.pad(a, (0, delay)).astype(np.float32)
    return a, b


def _max_window_rms(sig: np.ndarray, start: int, end: int, window: int) -> float:
    start = max(0, min(len(sig), start))
    end = max(start, min(len(sig), end))
    if end <= start:
        return 0.0
    x = sig[start:end]
    if len(x) <= window:
        return float(np.sqrt(np.maximum(np.mean(x ** 2), 1e-12)))
    vals = []
    step = max(1, window // 2)
    for pos in range(0, max(1, len(x) - window + 1), step):
        chunk = x[pos:pos + window]
        vals.append(float(np.sqrt(np.maximum(np.mean(chunk ** 2), 1e-12))))
    return max(vals) if vals else 0.0


def pair_samples_silent(
    a: np.ndarray,
    b: np.ndarray,
    start: int,
    end: int,
    sr: int,
    threshold: float,
    *,
    window_ms: float = 30.0,
) -> bool:
    start = max(0, min(len(a), start))
    end = max(start, min(len(a), end))
    if end <= start:
        return False
    window = max(1, int(round(sr * window_ms / 1000.0)))
    # Metadata VAD can miss tiny speech fragments; use waveform RMS as a hard veto.
    silence_threshold = max(0.006, float(threshold) * 0.70)
    return (
        _max_window_rms(a, start, end, window) <= silence_threshold
        and _max_window_rms(b, start, end, window) <= silence_threshold
    )


def channel_samples_silent(
    sig: np.ndarray,
    start: int,
    end: int,
    sr: int,
    threshold: float,
    *,
    window_ms: float = 30.0,
) -> bool:
    start = max(0, min(len(sig), start))
    end = max(start, min(len(sig), end))
    if end <= start:
        return False
    window = max(1, int(round(sr * window_ms / 1000.0)))
    silence_threshold = max(0.006, float(threshold) * 0.70)
    short_window = max(1, int(round(sr * 0.008)))
    burst_threshold = max(0.004, float(threshold) * 0.35)
    peak_threshold = max(0.010, float(threshold) * 0.80)
    x = np.asarray(sig[start:end], dtype=np.float32)
    return (
        _max_window_rms(sig, start, end, window) <= silence_threshold
        and _max_window_rms(sig, start, end, short_window) <= burst_threshold
        and (float(np.max(np.abs(x))) if len(x) else 0.0) <= peak_threshold
    )


def channel_silence_quality(sig: np.ndarray, start: int, end: int, sr: int) -> float:
    start = max(0, min(len(sig), start))
    end = max(start, min(len(sig), end))
    if end <= start:
        return float("inf")
    x = np.asarray(sig[start:end], dtype=np.float32)
    short_window = max(1, int(round(sr * 0.008)))
    return (
        float(np.sqrt(np.maximum(np.mean(x ** 2), 1e-12)))
        + 0.50 * _max_window_rms(sig, start, end, short_window)
        + 0.10 * float(np.max(np.abs(x)))
    )


def stereo_region_silent(stereo: np.ndarray, start: int, end: int, sr: int, threshold: float) -> bool:
    if stereo.ndim != 2 or stereo.shape[1] < 2:
        return False
    return pair_samples_silent(stereo[:, 0], stereo[:, 1], start, end, sr, threshold)


def vad_samples_silent(
    vad: np.ndarray | None,
    start: int,
    end: int,
    sr: int,
    frame_ms: float | None,
    *,
    guard_ms: float = 80.0,
) -> bool:
    if vad is None or frame_ms is None:
        return True
    start_ms = max(0.0, float(start) * 1000.0 / float(sr) - guard_ms)
    end_ms = max(start_ms, float(end) * 1000.0 / float(sr) + guard_ms)
    s = max(0, int(math.floor(start_ms / frame_ms)))
    e = min(len(vad), int(math.ceil(end_ms / frame_ms)))
    return bool(e <= s or vad[s:e].max() == 0)


def pair_vad_samples_silent(
    va: np.ndarray | None,
    vb: np.ndarray | None,
    start: int,
    end: int,
    sr: int,
    frame_ms: float | None,
    *,
    guard_ms: float = 80.0,
) -> bool:
    return (
        vad_samples_silent(va, start, end, sr, frame_ms, guard_ms=guard_ms)
        and vad_samples_silent(vb, start, end, sr, frame_ms, guard_ms=guard_ms)
    )


def _continuous_silent_room_tone(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    threshold: float,
    *,
    center: int,
    length: int,
    rng: random.Random,
    search_radius_s: float = 60.0,
    step_ms: float = 40.0,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
    frame_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one continuous clean room-tone span; avoid repeated short chunks."""
    need = max(1, int(length))
    total_len = min(len(a), len(b))
    if need > total_len:
        return None

    radius = int(round(sr * search_radius_s))
    local_lo = max(0, int(center) - radius)
    local_hi = min(total_len, int(center) + radius)
    ranges = [(local_lo, local_hi)]
    if local_hi - local_lo < total_len:
        ranges.append((0, total_len))

    step = max(1, int(round(sr * step_ms / 1000.0)))
    for lo, hi in ranges:
        if hi - lo < need:
            continue
        candidates: list[int] = []
        for pos in range(lo, hi - need + 1, step):
            if not pair_vad_samples_silent(va, vb, pos, pos + need, sr, frame_ms):
                continue
            if pair_samples_silent(a, b, pos, pos + need, sr, threshold):
                candidates.append(pos)
        if candidates:
            candidates.sort(key=lambda pos: abs((pos + need // 2) - int(center)))
            pool = candidates[: min(8, len(candidates))]
            pos = rng.choice(pool)
            return a[pos:pos + need].copy(), b[pos:pos + need].copy()
    return None


def _continuous_channel_room_tone(
    sig: np.ndarray,
    vad: np.ndarray | None,
    sr: int,
    threshold: float,
    *,
    length: int,
    rng: random.Random,
    preferred_ranges: list[tuple[int, int]],
    fallback_center: int,
    frame_ms: float | None = None,
    search_radius_s: float = 6.0,
    step_ms: float = 40.0,
) -> tuple[np.ndarray, str] | None:
    """Return a continuous single-channel silence bed near the requested region."""
    need = max(1, int(length))
    total_len = len(sig)
    if need > total_len:
        return None

    radius = int(round(sr * search_radius_s))
    ranges: list[tuple[int, int, str]] = []
    for lo, hi in preferred_ranges:
        lo = max(0, min(total_len, int(lo)))
        hi = max(lo, min(total_len, int(hi)))
        ranges.append((lo, hi, "preferred"))
    ranges.append((max(0, int(fallback_center) - radius), min(total_len, int(fallback_center) + radius), "nearby"))
    ranges.append((0, total_len, "global"))

    step = max(1, int(round(sr * step_ms / 1000.0)))
    seen_ranges: set[tuple[int, int, str]] = set()
    for lo, hi, label in ranges:
        key = (lo, hi, label)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        if hi - lo < need:
            continue
        candidates: list[tuple[float, int, int]] = []
        for pos in range(lo, hi - need + 1, step):
            if not vad_samples_silent(vad, pos, pos + need, sr, frame_ms):
                continue
            if channel_samples_silent(sig, pos, pos + need, sr, threshold):
                center_dist = abs((pos + need // 2) - fallback_center)
                candidates.append((channel_silence_quality(sig, pos, pos + need, sr), center_dist, pos))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            pos = candidates[0][2]
            return sig[pos:pos + need].copy(), label
    return None


def make_per_channel_response_room_tone(
    a: np.ndarray,
    b: np.ndarray,
    anchor: int,
    responder_start: int,
    length: int,
    sr: int,
    threshold: float,
    rng: random.Random,
    *,
    speaker_ch: str,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
    frame_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Build fill from each channel's own silence, not from mutual silence."""
    near = int(round(sr * 3.0))
    fallback = int(round(sr * 6.0))
    source_guard = int(round(sr * 0.20))
    responder_ch = "b" if speaker_ch == "a" else "a"

    def one(channel: str) -> tuple[np.ndarray, str] | None:
        sig = a if channel == "a" else b
        vad = va if channel == "a" else vb
        if channel == speaker_ch:
            # Current speaker is usually quiet after yielding the turn.
            ranges = [(anchor, min(len(sig), anchor + near)), (anchor, min(len(sig), anchor + fallback))]
            fallback_center = anchor
        elif channel == responder_ch:
            # Responder is usually quiet before their response while the other person is talking.
            responder_guarded_start = max(0, responder_start - source_guard)
            ranges = [
                (max(0, responder_guarded_start - near), responder_guarded_start),
                (max(0, responder_guarded_start - fallback), responder_guarded_start),
            ]
            fallback_center = responder_guarded_start
        else:
            ranges = [(max(0, anchor - fallback), min(len(sig), anchor + fallback))]
            fallback_center = anchor
        out = _continuous_channel_room_tone(
            sig,
            vad,
            sr,
            threshold,
            length=length,
            rng=rng,
            preferred_ranges=ranges,
            fallback_center=fallback_center,
            frame_ms=frame_ms,
        )
        if out is None:
            return None
        tone, source = out
        tone = _condition_room_tone(tone, threshold)
        return tone.astype(np.float32), source

    out_a = one("a")
    out_b = one("b")
    if out_a is None or out_b is None:
        return None
    tone_a, source_a = out_a
    tone_b, source_b = out_b
    source = f"per_channel_vad_filtered_silence:a={source_a},b={source_b}"
    return tone_a, tone_b, source


def _collect_room_tone_chunks(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    threshold: float,
    *,
    center: int,
    chunk_len: int,
    rng: random.Random,
    search_radius_s: float = 90.0,
    step_ms: float = 80.0,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
    frame_ms: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    total_len = min(len(a), len(b))
    chunk_len = max(1, min(int(chunk_len), total_len))
    radius = int(round(sr * search_radius_s))
    ranges = [
        (max(0, int(center) - radius), min(total_len, int(center) + radius)),
        (0, total_len),
    ]
    step = max(1, int(round(sr * step_ms / 1000.0)))
    found: list[tuple[int, np.ndarray, np.ndarray]] = []
    seen: set[int] = set()
    for lo, hi in ranges:
        if hi - lo < chunk_len:
            continue
        for pos in range(lo, hi - chunk_len + 1, step):
            if pos in seen:
                continue
            seen.add(pos)
            if not pair_vad_samples_silent(va, vb, pos, pos + chunk_len, sr, frame_ms):
                continue
            if pair_samples_silent(a, b, pos, pos + chunk_len, sr, threshold):
                found.append((abs((pos + chunk_len // 2) - int(center)), a[pos:pos + chunk_len].copy(), b[pos:pos + chunk_len].copy()))
    found.sort(key=lambda x: x[0])
    nearest = found[: min(32, len(found))]
    rng.shuffle(nearest)
    return [(ca, cb) for _, ca, cb in nearest]


def _overlap_add_room_tone(
    chunks: list[tuple[np.ndarray, np.ndarray]],
    length: int,
    sr: int,
    rng: random.Random,
    *,
    crossfade_ms: float = 60.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not chunks or length <= 0:
        return None
    crossfade = max(1, int(round(sr * crossfade_ms / 1000.0)))

    def add_one(out: np.ndarray | None, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if out is None or len(out) == 0:
            return chunk.copy()
        f = min(crossfade, len(out), len(chunk) // 2)
        if f <= 1:
            return np.concatenate([out, chunk]).astype(np.float32)
        fade_out, fade_in = _equal_power_curves(f)
        mixed = out[-f:] * fade_out + chunk[:f] * fade_in
        return np.concatenate([out[:-f], mixed, chunk[f:]]).astype(np.float32)

    out_a: np.ndarray | None = None
    out_b: np.ndarray | None = None
    i = 0
    while out_a is None or len(out_a) < length:
        ca, cb = chunks[i % len(chunks)]
        out_a = add_one(out_a, ca)
        out_b = add_one(out_b, cb)
        i += 1
        if i > max(4, len(chunks) * 4) and out_a is not None and len(out_a) < length:
            # Avoid an infinite loop on pathological tiny chunks.
            break
    if out_a is None or out_b is None:
        return None
    if len(out_a) < length:
        out_a = np.pad(out_a, (0, length - len(out_a)))
        out_b = np.pad(out_b, (0, length - len(out_b)))
    return out_a[:length].astype(np.float32), out_b[:length].astype(np.float32)


def _safe_rms(sig: np.ndarray) -> float:
    if len(sig) == 0:
        return 0.0
    x = np.asarray(sig, dtype=np.float32)
    return float(np.sqrt(np.maximum(np.mean(x ** 2), 1e-12)))


def _match_extended_tone_to_pre_context(
    tone: np.ndarray,
    ref: np.ndarray,
    start: int,
    sr: int,
    threshold: float,
    *,
    context_ms: float = 250.0,
) -> np.ndarray:
    tone = np.asarray(tone, dtype=np.float32).reshape(-1).copy()
    if len(tone) == 0:
        return tone
    ctx = max(1, int(round(sr * context_ms / 1000.0)))
    start = max(0, min(len(ref), int(start)))
    ref_seg = np.asarray(ref[max(0, start - ctx):start], dtype=np.float32)
    ref_rms = _safe_rms(ref_seg)
    tone_rms = _safe_rms(tone)
    # Fixed gain only: align with preceding ambience without creating a time-varying pump.
    ceiling = max(0.0025, float(threshold) * 0.22)
    target = min(ceiling, max(0.00020, ref_rms * 0.85))
    gain = min(3.0, max(0.35, target / max(tone_rms, 1e-8)))
    return (tone * gain).astype(np.float32)


def _highpassed_room_floor(length: int, sr: int, rng: random.Random) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    np_rng = np.random.default_rng(rng.randrange(0, 2**32))
    noise = np_rng.normal(0.0, 1.0, size=int(length)).astype(np.float32)
    # Remove slow/low-frequency structure; this only restores the missing broad high-frequency bed.
    win = max(3, int(round(sr * 0.006)))
    kernel = np.ones((win,), dtype=np.float32) / float(win)
    low = np.convolve(noise, kernel, mode="same").astype(np.float32)
    hp = noise - low
    hp -= float(np.mean(hp))
    return hp.astype(np.float32)


def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sigma <= 0 or len(x) == 0:
        return x.astype(np.float32)
    radius = max(1, int(round(4.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / float(sigma)) ** 2).astype(np.float32)
    k /= max(float(k.sum()), 1e-8)
    return np.convolve(x, k, mode="same").astype(np.float32)


def _sampled_smoothed_local_silence_pair(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    threshold: float,
    rng: random.Random,
    *,
    center: int,
    length: int,
    context_start: int,
    context_end: int,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
    frame_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Build fill by sampling nearby real silence chunks and smoothing chunk joins."""
    length = int(length)
    if length <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, "empty"
    total_len = min(len(a), len(b))
    center = max(0, min(total_len, int(center)))
    context_start = max(0, min(total_len, int(context_start)))
    context_end = max(context_start, min(total_len, int(context_end)))

    chunk_len = max(int(round(sr * 0.28)), min(length, int(round(sr * 0.55))))
    step = max(1, int(round(sr * 0.04)))
    search = int(round(sr * 4.0))
    ranges = [
        (context_start, context_end),
        (max(0, center - search), min(total_len, center + search)),
    ]
    strict_candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    relaxed_candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    seen: set[int] = set()
    for lo, hi in ranges:
        if hi - lo < chunk_len:
            continue
        for pos in range(lo, hi - chunk_len + 1, step):
            if pos in seen:
                continue
            seen.add(pos)
            if not pair_samples_silent(a, b, pos, pos + chunk_len, sr, threshold):
                continue
            item = (abs((pos + chunk_len // 2) - center), a[pos:pos + chunk_len].copy(), b[pos:pos + chunk_len].copy())
            if pair_vad_samples_silent(va, vb, pos, pos + chunk_len, sr, frame_ms, guard_ms=40.0):
                strict_candidates.append(item)
            else:
                relaxed_candidates.append(item)
    candidates = strict_candidates or relaxed_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    pool = candidates[: min(48, len(candidates))]
    rng.shuffle(pool)
    chunks = [(ca, cb) for _, ca, cb in pool]
    bed = _overlap_add_room_tone(chunks, length, sr, rng, crossfade_ms=120.0)
    if bed is None:
        return None

    def condition(tone: np.ndarray, ref: np.ndarray) -> np.ndarray:
        tone = np.asarray(tone, dtype=np.float32).copy()
        tone -= float(np.mean(tone))
        def quiet_context_target() -> float:
            win = max(1, int(round(sr * 0.030)))
            step = max(1, int(round(sr * 0.010)))
            vals: list[float] = []
            spans = [
                (context_start, context_end),
                (max(0, center - int(round(sr * 2.0))), min(len(ref), center + int(round(sr * 2.0)))),
            ]
            veto = max(0.006, float(threshold) * 0.70)
            for lo, hi in spans:
                lo = max(0, min(len(ref), int(lo)))
                hi = max(lo, min(len(ref), int(hi)))
                if hi - lo < win:
                    continue
                for pos in range(lo, hi - win + 1, step):
                    chunk = ref[pos:pos + win]
                    r = _safe_rms(chunk)
                    if r <= veto:
                        vals.append(r)
            if vals:
                quiet = float(np.percentile(np.asarray(vals, dtype=np.float32), 35.0))
            else:
                quiet = _safe_rms(ref[max(0, context_start):max(context_start, context_end)])
            upper = max(8e-5, float(threshold) * 0.025)
            return max(2e-5, min(float(quiet) * 0.85, upper))

        target = quiet_context_target()
        gain = target / max(_safe_rms(tone), 1e-8)
        gain = min(2.0, max(0.10, gain))
        tone *= gain
        # Smooth only the slow amplitude contour, not the waveform spectrum.
        win = max(1, int(round(sr * 0.030)))
        if len(tone) > win * 2:
            env = np.sqrt(np.convolve(tone ** 2, np.ones((win,), dtype=np.float32) / float(win), mode="same") + 1e-12)
            smooth_env = _gaussian_smooth_1d(env, sigma=max(1.0, win / 3.0))
            ratio = smooth_env / np.maximum(env, max(target * 0.35, 1e-6))
            tone *= np.clip(ratio, 0.65, 1.35)
            final_gain = target / max(_safe_rms(tone), 1e-8)
            tone *= min(1.8, max(0.55, final_gain))
        return tone.astype(np.float32)

    return condition(bed[0], a), condition(bed[1], b), "sampled_smoothed_local_silence"


def _spectral_floor_from_context(
    sig: np.ndarray,
    length: int,
    sr: int,
    threshold: float,
    rng: random.Random,
    *,
    context_start: int,
    context_end: int,
) -> np.ndarray | None:
    """Synthesize a low room floor with the local gap's spectral color."""
    length = int(length)
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    context_start = max(0, min(len(sig), int(context_start)))
    context_end = max(context_start, min(len(sig), int(context_end)))
    ref = np.asarray(sig[context_start:context_end], dtype=np.float32).reshape(-1)
    if len(ref) < max(16, int(round(sr * 0.04))):
        return None
    ref = ref - float(np.mean(ref))
    ref_rms = _safe_rms(ref)

    n_fft = 2048 if sr >= 24000 else 1024
    hop = max(1, n_fft // 4)
    if len(ref) < n_fft:
        ref = np.pad(ref, (0, n_fft - len(ref)), mode="reflect" if len(ref) > 1 else "constant")
    window = np.hanning(n_fft).astype(np.float32)
    mags: list[np.ndarray] = []
    for pos in range(0, len(ref) - n_fft + 1, hop):
        mags.append(np.abs(np.fft.rfft(ref[pos:pos + n_fft] * window)).astype(np.float32))
    if not mags:
        return None
    mag = np.median(np.stack(mags, axis=0), axis=0).astype(np.float32)
    if len(mag) >= 7:
        kernel = np.ones((7,), dtype=np.float32) / 7.0
        mag = np.convolve(mag, kernel, mode="same").astype(np.float32)
    if not np.isfinite(mag).all() or float(np.max(mag)) <= 1e-10:
        return None

    np_rng = np.random.default_rng(rng.randrange(0, 2**32))
    synth_len = int(length) + n_fft
    n_frames = int(math.ceil(max(1, synth_len - n_fft) / float(hop))) + 1
    out = np.zeros((n_frames * hop + n_fft,), dtype=np.float32)
    norm = np.zeros_like(out)
    for i in range(n_frames):
        phase = np_rng.uniform(0.0, 2.0 * math.pi, size=mag.shape).astype(np.float32)
        phase[0] = 0.0
        if len(phase) > 1:
            phase[-1] = 0.0
        spec = mag * np.exp(1j * phase)
        frame = np.fft.irfft(spec, n=n_fft).astype(np.float32)
        pos = i * hop
        out[pos:pos + n_fft] += frame * window
        norm[pos:pos + n_fft] += window ** 2
    valid = norm > 1e-8
    out[valid] /= norm[valid]
    out = out[:length].astype(np.float32)
    out -= float(np.mean(out))

    # Keep the spectrogram floor visible, but under speech and close to local ambience.
    target = min(max(0.0022, ref_rms * 2.0), max(0.0040, float(threshold) * 0.25))
    out_rms = _safe_rms(out)
    out *= target / max(out_rms, 1e-8)

    # Some channels have a gap reference with almost no high-frequency bins, which
    # creates an obvious dark vertical band in the spectrogram. Add a very soft
    # high-passed floor, then renormalize so it fills the band without sounding loud.
    hp = _highpassed_room_floor(length, sr, rng)
    hp_rms = _safe_rms(hp)
    if hp_rms > 0.0:
        out = out + hp * ((target * 0.32) / max(hp_rms, 1e-8))
        out *= target / max(_safe_rms(out), 1e-8)
    return out.astype(np.float32)


def _spectral_gap_floor_pair(
    a: np.ndarray,
    b: np.ndarray,
    length: int,
    sr: int,
    threshold: float,
    rng: random.Random,
    *,
    context_start: int,
    context_end: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    fa = _spectral_floor_from_context(
        a, length, sr, threshold, rng, context_start=context_start, context_end=context_end
    )
    fb = _spectral_floor_from_context(
        b, length, sr, threshold, rng, context_start=context_start, context_end=context_end
    )
    if fa is None or fb is None:
        return None
    return fa, fb, "spectral_gap_floor_bridge"


def _find_preceding_silence_start(
    a: np.ndarray,
    b: np.ndarray,
    start: int,
    sr: int,
    threshold: float,
    *,
    max_back_s: float = 1.5,
    step_ms: float = 20.0,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
    frame_ms: float | None = None,
) -> int:
    start = max(0, min(len(a), int(start)))
    lo = max(0, start - int(round(sr * max_back_s)))
    step = max(1, int(round(sr * step_ms / 1000.0)))
    best = start
    for pos in range(start - step, lo - 1, -step):
        if not pair_vad_samples_silent(va, vb, pos, start, sr, frame_ms):
            break
        if not pair_samples_silent(a, b, pos, start, sr, threshold):
            break
        best = pos
    return best


def _extend_preceding_room_tone_pair(
    a: np.ndarray,
    b: np.ndarray,
    start: int,
    length: int,
    sr: int,
    threshold: float,
    rng: random.Random,
    *,
    source_start: int | None = None,
    min_source_ms: float = 80.0,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
    frame_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Extend the immediately preceding silence/room tone instead of importing new ambience."""
    if length <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, "empty"
    start = max(0, min(len(a), int(start)))
    if source_start is None:
        src_start = _find_preceding_silence_start(
            a, b, start, sr, threshold, va=va, vb=vb, frame_ms=frame_ms
        )
    else:
        src_start = max(0, min(start, int(source_start)))
        if not pair_vad_samples_silent(va, vb, src_start, start, sr, frame_ms) or not pair_samples_silent(
            a, b, src_start, start, sr, threshold
        ):
            src_start = _find_preceding_silence_start(
                a, b, start, sr, threshold, va=va, vb=vb, frame_ms=frame_ms
            )

    min_source = max(1, int(round(sr * min_source_ms / 1000.0)))
    if start - src_start < min_source:
        src_start = max(0, start - min_source)
        if not pair_vad_samples_silent(va, vb, src_start, start, sr, frame_ms) or not pair_samples_silent(
            a, b, src_start, start, sr, threshold
        ):
            return None

    src_a = np.asarray(a[src_start:start], dtype=np.float32).copy()
    src_b = np.asarray(b[src_start:start], dtype=np.float32).copy()
    if len(src_a) == 0 or len(src_b) == 0:
        return None
    bed = _overlap_add_room_tone([(src_a, src_b)], length, sr, rng, crossfade_ms=80.0)
    if bed is None:
        return None
    bed_a = _match_extended_tone_to_pre_context(bed[0], a, start, sr, threshold)
    bed_b = _match_extended_tone_to_pre_context(bed[1], b, start, sr, threshold)
    return bed_a, bed_b, "extended_preceding_mutual_silence"


def _condition_room_tone(tone: np.ndarray, threshold: float) -> np.ndarray:
    tone = np.asarray(tone, dtype=np.float32).reshape(-1).copy()
    if len(tone) == 0:
        return tone
    # Keep copied room tone mostly intact; only remove DC and prevent rare loud beds.
    tone -= float(np.mean(tone))
    rms = float(np.sqrt(np.maximum(np.mean(tone ** 2), 1e-12)))
    soft_ceiling = max(0.0025, float(threshold) * 0.25)
    if rms > soft_ceiling:
        tone *= soft_ceiling / max(rms, 1e-8)
    return tone.astype(np.float32)

def _shaped_room_floor(length: int, sr: int, threshold: float, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z
    np_rng = np.random.default_rng(rng.randrange(0, 2**32))
    scale = max(0.00035, float(threshold) * 0.04)
    outs = []
    for _ in range(2):
        noise = np_rng.normal(0.0, 1.0, size=length).astype(np.float32)
        win = max(3, int(round(sr * 0.002)))
        kernel = np.ones((win,), dtype=np.float32) / float(win)
        shaped = np.convolve(noise, kernel, mode="same").astype(np.float32)
        rms = float(np.sqrt(np.maximum(np.mean(shaped ** 2), 1e-12)))
        shaped *= scale / max(rms, 1e-8)
        outs.append(shaped.astype(np.float32))
    return outs[0], outs[1]


def make_room_tone_pair(
    a: np.ndarray,
    b: np.ndarray,
    start: int,
    length: int,
    sr: int,
    threshold: float,
    rng: random.Random,
    *,
    va: np.ndarray | None = None,
    vb: np.ndarray | None = None,
    frame_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if length <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, "empty"
    use_vad_filter = va is not None and vb is not None and frame_ms is not None

    tone = _continuous_silent_room_tone(
        a,
        b,
        sr,
        threshold,
        center=start,
        length=length,
        rng=rng,
        va=va,
        vb=vb,
        frame_ms=frame_ms,
    )
    if tone is not None:
        tone_a, tone_b = tone
        source = "continuous_vad_filtered_mutual_silence" if use_vad_filter else "continuous_mutual_silence"
    else:
        chunk_len = max(int(round(sr * 0.50)), min(length, int(round(sr * 0.80))))
        chunks = _collect_room_tone_chunks(
            a,
            b,
            sr,
            threshold,
            center=start,
            chunk_len=chunk_len,
            rng=rng,
            va=va,
            vb=vb,
            frame_ms=frame_ms,
        )
        bed = _overlap_add_room_tone(chunks, length, sr, rng)
        if bed is not None:
            tone_a, tone_b = bed
            source = "overlapadd_vad_filtered_mutual_silence" if use_vad_filter else "overlapadd_mutual_silence"
        else:
            tone_a, tone_b = _shaped_room_floor(length, sr, threshold, rng)
            source = "shaped_floor_no_clean_vad_room_tone" if use_vad_filter else "shaped_floor_no_clean_room_tone"

    tone_a = _condition_room_tone(tone_a, threshold)
    tone_b = _condition_room_tone(tone_b, threshold)
    return tone_a.astype(np.float32), tone_b.astype(np.float32), source

def _fade_len_samples(sr: int, fade_ms: float) -> int:
    return max(0, int(round(float(sr) * float(fade_ms) / 1000.0)))


def _equal_power_curves(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z
    t = np.linspace(0.0, math.pi / 2.0, n, dtype=np.float32)
    return np.cos(t).astype(np.float32), np.sin(t).astype(np.float32)


def _fit_fill_channel(fill: np.ndarray | None, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if fill is None:
        return np.zeros((n,), dtype=np.float32)
    x = np.asarray(fill, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return np.zeros((n,), dtype=np.float32)
    if len(x) >= n:
        return x[:n].copy()
    out = np.zeros((n,), dtype=np.float32)
    out[:len(x)] = x
    return out


def _insert_with_crossfade(
    sig: np.ndarray,
    start: int,
    delay: int,
    fill: np.ndarray | None,
    *,
    sr: int,
    fade_ms: float = 10.0,
) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if delay <= 0:
        return sig.astype(np.float32, copy=True)
    start = max(0, min(len(sig), int(start)))
    fade = _fade_len_samples(sr, fade_ms)
    left_fade = min(fade, start)
    right_fade = min(fade, len(sig) - start)
    fill_full = _fit_fill_channel(fill, int(delay) + left_fade + right_fade)

    left_fill = fill_full[:left_fade]
    mid_fill = fill_full[left_fade:left_fade + int(delay)]
    right_fill = fill_full[left_fade + int(delay):left_fade + int(delay) + right_fade]

    parts: list[np.ndarray] = []
    if left_fade > 0:
        fade_out, fade_in = _equal_power_curves(left_fade)
        parts.append(sig[:start - left_fade])
        parts.append(sig[start - left_fade:start] * fade_out + left_fill * fade_in)
    else:
        parts.append(sig[:start])

    parts.append(mid_fill)

    if right_fade > 0:
        fade_out, fade_in = _equal_power_curves(right_fade)
        parts.append(right_fill * fade_out + sig[start:start + right_fade] * fade_in)
        parts.append(sig[start + right_fade:])
    else:
        parts.append(sig[start:])

    out = np.concatenate(parts).astype(np.float32)
    assert len(out) == len(sig) + int(delay)
    return out


def _delete_with_boundary_fade(
    sig: np.ndarray,
    cut_start: int,
    cut_len: int,
    *,
    sr: int,
    fade_ms: float = 10.0,
) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if cut_len <= 0:
        return sig.astype(np.float32, copy=True)
    cut_start = max(0, min(len(sig), int(cut_start)))
    cut_end = max(cut_start, min(len(sig), cut_start + int(cut_len)))
    if cut_end <= cut_start:
        return sig.astype(np.float32, copy=True)

    pre = sig[:cut_start].copy()
    post = sig[cut_end:].copy()
    fade = _fade_len_samples(sr, fade_ms)
    left_fade = min(fade, len(pre))
    right_fade = min(fade, len(post))
    if left_fade > 0:
        fade_out, _ = _equal_power_curves(left_fade)
        pre[-left_fade:] *= fade_out
    if right_fade > 0:
        _, fade_in = _equal_power_curves(right_fade)
        post[:right_fade] *= fade_in
    out = np.concatenate([pre, post]).astype(np.float32)
    assert len(out) == len(sig) - (cut_end - cut_start)
    return out


def shift_pair_later(
    a: np.ndarray,
    b: np.ndarray,
    start: int,
    delay: int,
    fill: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    sr: int,
    fade_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert room tone/silence in both channels; output duration increases by delay."""
    fa, fb = fill if fill is not None else (None, None)
    return (
        _insert_with_crossfade(a, start, delay, fa, sr=sr, fade_ms=fade_ms),
        _insert_with_crossfade(b, start, delay, fb, sr=sr, fade_ms=fade_ms),
    )


def shift_pair_earlier(
    a: np.ndarray,
    b: np.ndarray,
    cut_start: int,
    cut_len: int,
    *,
    sr: int,
    fade_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a span in both channels; output duration decreases by the cut length."""
    return (
        _delete_with_boundary_fade(a, cut_start, cut_len, sr=sr, fade_ms=fade_ms),
        _delete_with_boundary_fade(b, cut_start, cut_len, sr=sr, fade_ms=fade_ms),
    )


def shift_channel_earlier_with_fill(sig: np.ndarray, cut_start: int, cut_end: int, fill: np.ndarray) -> np.ndarray:
    cut_start = max(0, min(len(sig), cut_start))
    cut_end = max(cut_start, min(len(sig), cut_end))
    if cut_end <= cut_start:
        return sig
    need = cut_end - cut_start
    fill = np.asarray(fill, dtype=np.float32)[:need]
    if len(fill) < need:
        fill = np.pad(fill, (0, need - len(fill)))
    return np.concatenate([sig[:cut_start], sig[cut_end:], fill]).astype(np.float32)

def next_segment_start_frame(segs: list[Segment], seg: Segment) -> int | None:
    later = [s.start for s in segs if s.start > seg.end]
    if not later:
        return None
    return min(later)


def segment_word_count(seg: Segment) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", seg.text or ""))


BACKCHANNEL_PHRASES = {
    "mm", "mhm", "hm", "hmm", "um", "uh", "uh huh", "uh-huh", "yeah", "yea", "yep", "yes",
    "right", "okay", "ok", "sure", "true", "exactly", "wow", "oh", "oh wow",
    "hell yeah", "yeah yeah", "yeah right",
}


def normalized_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_backchannel_like(seg: Segment, max_frames: int = 14, max_words: int = 4) -> bool:
    raw = seg.text or ""
    text = normalized_text(raw)
    n_words = segment_word_count(seg)
    if "?" in raw or any(tok in text.split() for tok in ("what", "why", "how", "when", "where", "who")):
        return False
    if not text:
        return seg.dur <= max_frames
    if text in BACKCHANNEL_PHRASES:
        return True
    if seg.dur <= max_frames and n_words <= max_words:
        starts = ("yeah", "yep", "yes", "right", "ok", "okay", "mm", "hm", "uh", "oh")
        return text.startswith(starts)
    return False


def is_substantive_turn(seg: Segment, min_frames: int = 12, min_words: int = 4) -> bool:
    if seg.dur < min_frames:
        return False
    if not (seg.text or "").strip():
        return True
    if segment_word_count(seg) < min_words:
        return False
    if is_backchannel_like(seg):
        return False
    return True


def find_shift_points(
    segs_a: list[Segment],
    segs_b: list[Segment],
    max_gap_frames: int = 50,
    require_substantive: bool = True,
) -> list[tuple[Segment, Segment]]:
    pairs = []
    for sa in segs_a:
        if require_substantive and not is_substantive_turn(sa):
            continue
        for sb in segs_b:
            if require_substantive and not is_substantive_turn(sb):
                continue
            if sb.start > sa.end and (sb.start - sa.end) <= max_gap_frames:
                pairs.append((sa, sb))
                break
    return pairs


def find_shift_points_bidir(
    segs_a: list[Segment],
    segs_b: list[Segment],
    max_gap_frames: int = 50,
) -> list[tuple[str, Segment, Segment]]:
    """
    Returns tuples of (speaker_channel, speaker_seg, responder_seg).

    - ("a", seg_a, seg_b) means A speaks first, then B responds
    - ("b", seg_b, seg_a) means B speaks first, then A responds
    """
    pairs: list[tuple[str, Segment, Segment]] = []
    for sa, sb in find_shift_points(segs_a, segs_b, max_gap_frames=max_gap_frames):
        pairs.append(("a", sa, sb))
    for sb, sa in find_shift_points(segs_b, segs_a, max_gap_frames=max_gap_frames):
        pairs.append(("b", sb, sa))
    return pairs


def find_hold_regions_bidir(
    segs_a: list[Segment],
    segs_b: list[Segment],
) -> list[tuple[str, Segment, Segment]]:
    """
    Returns tuples of (hold_channel, curr_seg, next_same_speaker_seg)
    where the other speaker stays silent in the gap.
    """
    out: list[tuple[str, Segment, Segment]] = []

    for i in range(len(segs_a) - 1):
        curr = segs_a[i]
        nxt = segs_a[i + 1]
        gap = nxt.start - curr.end - 1
        if gap <= 0:
            continue
        silent = True
        for sb in segs_b:
            if not (sb.end < curr.end + 1 or sb.start > nxt.start - 1):
                silent = False
                break
        if silent:
            out.append(("a", curr, nxt))

    for i in range(len(segs_b) - 1):
        curr = segs_b[i]
        nxt = segs_b[i + 1]
        gap = nxt.start - curr.end - 1
        if gap <= 0:
            continue
        silent = True
        for sa in segs_a:
            if not (sa.end < curr.end + 1 or sa.start > nxt.start - 1):
                silent = False
                break
        if silent:
            out.append(("b", curr, nxt))

    return out


def _active(vad: np.ndarray, start: int, end: int) -> bool:
    start = max(0, min(len(vad), start))
    end = max(start, min(len(vad), end))
    return bool(end > start and vad[start:end].max() > 0)


def _silent(vad: np.ndarray, start: int, end: int) -> bool:
    start = max(0, min(len(vad), start))
    end = max(start, min(len(vad), end))
    return bool(end <= start or vad[start:end].max() == 0)


def _mutual_silence(va: np.ndarray, vb: np.ndarray, start: int, end: int) -> bool:
    return _silent(va, start, end) and _silent(vb, start, end)


def _only_active(active_vad: np.ndarray, other_vad: np.ndarray, start: int, end: int) -> bool:
    start_i = int(start)
    end_i = int(end)
    if start_i < 0 or end_i > len(active_vad) or end_i > len(other_vad) or end_i <= start_i:
        return False
    return bool(active_vad[start_i:end_i].min() > 0 and other_vad[start_i:end_i].max() == 0)


def _proper_transition(
    pre_seg: Segment,
    post_seg: Segment,
    pre_vad: np.ndarray,
    post_vad: np.ndarray,
    *,
    frame_ms: float,
    pre_offset_ms: float = 1000.0,
    post_onset_ms: float = 1000.0,
    eval_start_ms: float = 50.0,
    eval_dur_ms: float = 100.0,
    max_gap_ms: float | None = None,
) -> bool:
    if post_seg.start <= pre_seg.end:
        return False
    pre_frames = max(1, int(math.ceil(pre_offset_ms / frame_ms)))
    post_frames = max(1, int(math.ceil(post_onset_ms / frame_ms)))
    eval_frames = max(1, int(math.ceil((eval_start_ms + eval_dur_ms) / frame_ms)))
    silence_start = pre_seg.end + 1
    silence_end = post_seg.start
    if silence_end - silence_start < eval_frames:
        return False
    if max_gap_ms is not None and (silence_end - silence_start) * float(frame_ms) > float(max_gap_ms):
        return False
    if not _mutual_silence(pre_vad, post_vad, silence_start, silence_end):
        return False
    if not _only_active(pre_vad, post_vad, pre_seg.end + 1 - pre_frames, pre_seg.end + 1):
        return False
    if not _only_active(post_vad, pre_vad, post_seg.start, post_seg.start + post_frames):
        return False
    return True


def _proper_hold_transition(
    curr_seg: Segment,
    next_seg: Segment,
    speaker_vad: np.ndarray,
    other_vad: np.ndarray,
    *,
    frame_ms: float,
    pre_offset_ms: float = 1000.0,
    post_onset_ms: float = 1000.0,
    eval_start_ms: float = 50.0,
    eval_dur_ms: float = 100.0,
) -> bool:
    if next_seg.start <= curr_seg.end:
        return False
    pre_frames = max(1, int(math.ceil(pre_offset_ms / frame_ms)))
    post_frames = max(1, int(math.ceil(post_onset_ms / frame_ms)))
    eval_frames = max(1, int(math.ceil((eval_start_ms + eval_dur_ms) / frame_ms)))
    silence_start = curr_seg.end + 1
    silence_end = next_seg.start
    if silence_end - silence_start < eval_frames:
        return False
    if not _mutual_silence(speaker_vad, other_vad, silence_start, silence_end):
        return False
    if not _only_active(speaker_vad, other_vad, curr_seg.end + 1 - pre_frames, curr_seg.end + 1):
        return False
    if not _only_active(speaker_vad, other_vad, next_seg.start, next_seg.start + post_frames):
        return False
    return True


def find_proper_shift_points_bidir(
    segs_a: list[Segment],
    segs_b: list[Segment],
    va: np.ndarray,
    vb: np.ndarray,
    *,
    frame_ms: float,
    max_gap_ms: float | None = None,
) -> list[tuple[str, Segment, Segment]]:
    out: list[tuple[str, Segment, Segment]] = []
    for sa in segs_a:
        if not is_substantive_turn(sa):
            continue
        for sb in segs_b:
            if is_substantive_turn(sb) and _proper_transition(sa, sb, va, vb, frame_ms=frame_ms, max_gap_ms=max_gap_ms):
                out.append(("a", sa, sb))
                break
    for sb in segs_b:
        if not is_substantive_turn(sb):
            continue
        for sa in segs_a:
            if is_substantive_turn(sa) and _proper_transition(sb, sa, vb, va, frame_ms=frame_ms, max_gap_ms=max_gap_ms):
                out.append(("b", sb, sa))
                break
    return out


def find_active_alone_shift_points_bidir(
    segs_a: list[Segment],
    segs_b: list[Segment],
    va: np.ndarray,
    vb: np.ndarray,
    *,
    frame_ms: float,
    pre_offset_ms: float = 1000.0,
    post_onset_ms: float = 1000.0,
    max_overlap_ms: float = 200.0,
    max_gap_ms: float | None = None,
    min_pre_duration_ms: float = 1000.0,
    min_post_duration_ms: float = 1000.0,
) -> list[tuple[str, Segment, Segment]]:
    """Find A->B transitions using active-alone guards but no silence-gap requirement."""
    pre_frames = max(1, int(math.ceil(pre_offset_ms / frame_ms)))
    post_frames = max(1, int(math.ceil(post_onset_ms / frame_ms)))
    max_overlap_frames = max(0, int(math.ceil(max_overlap_ms / frame_ms)))
    max_gap_frames = None if max_gap_ms is None else max(0, int(math.ceil(max_gap_ms / frame_ms)))
    min_pre_frames = max(1, int(math.ceil(min_pre_duration_ms / frame_ms)))
    min_post_frames = max(1, int(math.ceil(min_post_duration_ms / frame_ms)))
    out: list[tuple[str, Segment, Segment]] = []

    def ok(pre: Segment, post: Segment, pre_vad: np.ndarray, post_vad: np.ndarray) -> bool:
        if pre.dur < min_pre_frames or post.dur < min_post_frames:
            return False
        gap_frames = int(post.start) - int(pre.end)
        if gap_frames < -max_overlap_frames:
            return False
        if max_gap_frames is not None and gap_frames > max_gap_frames:
            return False
        # We do not require a minimum gap, but if there is a positive gap it
        # must be a real mutual-silence gap. This prevents skipping over short
        # intervening turns such as "Were you by yourself?" / "Yeah".
        if gap_frames > 0 and not _mutual_silence(pre_vad, post_vad, pre.end + 1, post.start):
            return False
        if not _only_active(pre_vad, post_vad, pre.end + 1 - pre_frames, pre.end + 1):
            return False
        if not _only_active(post_vad, pre_vad, post.start, post.start + post_frames):
            return False
        return True

    for sa in segs_a:
        for sb in segs_b:
            if ok(sa, sb, va, vb):
                out.append(("a", sa, sb))
                break
    for sb in segs_b:
        for sa in segs_a:
            if ok(sb, sa, vb, va):
                out.append(("b", sb, sa))
                break
    return out


def find_proper_hold_regions_bidir(
    segs_a: list[Segment],
    segs_b: list[Segment],
    va: np.ndarray,
    vb: np.ndarray,
    *,
    frame_ms: float,
) -> list[tuple[str, Segment, Segment]]:
    out: list[tuple[str, Segment, Segment]] = []
    for i in range(len(segs_a) - 1):
        curr, nxt = segs_a[i], segs_a[i + 1]
        if _proper_hold_transition(curr, nxt, va, vb, frame_ms=frame_ms):
            out.append(("a", curr, nxt))
    for i in range(len(segs_b) - 1):
        curr, nxt = segs_b[i], segs_b[i + 1]
        if _proper_hold_transition(curr, nxt, vb, va, frame_ms=frame_ms):
            out.append(("b", curr, nxt))
    return out


def _next_active_channel_after(
    segs_a: list[Segment],
    segs_b: list[Segment],
    frame: int,
) -> tuple[str, Segment] | None:
    cands: list[tuple[int, str, Segment]] = []
    for seg in segs_a:
        if seg.start > frame and is_substantive_turn(seg):
            cands.append((seg.start, "a", seg))
    for seg in segs_b:
        if seg.start > frame and is_substantive_turn(seg):
            cands.append((seg.start, "b", seg))
    if not cands:
        return None
    _, ch, seg = min(cands, key=lambda x: x[0])
    return ch, seg


def find_shift_then_original_speaker_returns(
    segs_a: list[Segment],
    segs_b: list[Segment],
    va: np.ndarray,
    vb: np.ndarray,
    *,
    frame_ms: float,
    max_gap_ms: float | None = None,
    require_return_proper: bool = True,
) -> list[tuple[str, Segment, Segment, Segment]]:
    out: list[tuple[str, Segment, Segment, Segment]] = []
    for speaker_ch, speaker_seg, responder_seg in find_proper_shift_points_bidir(segs_a, segs_b, va, vb, frame_ms=frame_ms, max_gap_ms=max_gap_ms):
        nxt = _next_active_channel_after(segs_a, segs_b, responder_seg.end)
        if nxt is None:
            continue
        next_ch, next_seg = nxt
        if next_ch != speaker_ch:
            continue
        if require_return_proper:
            if speaker_ch == "a":
                return_is_proper = _proper_transition(responder_seg, next_seg, vb, va, frame_ms=frame_ms, max_gap_ms=max_gap_ms)
            else:
                return_is_proper = _proper_transition(responder_seg, next_seg, va, vb, frame_ms=frame_ms, max_gap_ms=max_gap_ms)
            if not return_is_proper:
                continue
        out.append((speaker_ch, speaker_seg, responder_seg, next_seg))
    return out


def find_active_alone_shift_then_original_speaker_returns(
    segs_a: list[Segment],
    segs_b: list[Segment],
    va: np.ndarray,
    vb: np.ndarray,
    *,
    frame_ms: float,
    max_gap_ms: float | None = None,
    max_overlap_ms: float = 0.0,
    min_mutual_silence_ms: float = 100.0,
) -> list[tuple[str, Segment, Segment, Segment]]:
    """Find A->B->A turns using active-alone guards without VAP silence-window constraints."""
    out: list[tuple[str, Segment, Segment, Segment]] = []
    post_frames = max(1, int(math.ceil(1000.0 / frame_ms)))
    min_silence_frames = max(1, int(math.ceil(min_mutual_silence_ms / frame_ms)))
    shifts = find_active_alone_shift_points_bidir(
        segs_a,
        segs_b,
        va,
        vb,
        frame_ms=frame_ms,
        max_overlap_ms=max_overlap_ms,
        max_gap_ms=max_gap_ms,
        min_pre_duration_ms=1000.0,
        min_post_duration_ms=1000.0,
    )
    for speaker_ch, speaker_seg, responder_seg in shifts:
        if responder_seg.start - speaker_seg.end < min_silence_frames:
            continue
        nxt = _next_active_channel_after(segs_a, segs_b, responder_seg.end)
        if nxt is None:
            continue
        next_ch, next_seg = nxt
        if next_ch != speaker_ch:
            continue
        if speaker_ch == "a":
            if not _only_active(va, vb, next_seg.start, next_seg.start + post_frames):
                continue
        else:
            if not _only_active(vb, va, next_seg.start, next_seg.start + post_frames):
                continue
        out.append((speaker_ch, speaker_seg, responder_seg, next_seg))
    return out


def find_isolated_backchannels(
    listener_segs: list[Segment],
    listener_vad: np.ndarray,
    speaker_vad: np.ndarray,
    *,
    frame_ms: float,
    bc_duration_ms: float = 1000.0,
    pre_silence_ms: float = 1000.0,
    post_silence_ms: float = 2000.0,
) -> list[Segment]:
    max_bc_frames = max(1, int(math.ceil(bc_duration_ms / frame_ms)))
    pre_frames = max(1, int(math.ceil(pre_silence_ms / frame_ms)))
    post_frames = max(1, int(math.ceil(post_silence_ms / frame_ms)))
    out: list[Segment] = []
    for seg in listener_segs:
        if seg.dur > max_bc_frames or not is_backchannel_like(seg, max_frames=max_bc_frames):
            continue
        if not _silent(listener_vad, seg.start - pre_frames, seg.start):
            continue
        if not _silent(listener_vad, seg.end + 1, seg.end + 1 + post_frames):
            continue
        if not _active(speaker_vad, seg.start - pre_frames, seg.start):
            continue
        if not _active(speaker_vad, seg.start, seg.end + 1):
            continue
        out.append(seg)
    return out


def find_short_backchannels(listener_segs: list[Segment], speaker_vad: np.ndarray, max_bc_frames: int = 14) -> list[Segment]:
    out = []
    for seg in listener_segs:
        overlap = speaker_vad[seg.start:seg.end + 1]
        if seg.dur <= max_bc_frames and overlap.size and overlap.max() > 0 and is_backchannel_like(seg, max_frames=max_bc_frames):
            out.append(seg)
    return out


def pick_spaced_frames(
    frames: list[int] | np.ndarray,
    *,
    count: int,
    min_gap_frames: int,
    rng: random.Random,
) -> list[int]:
    """Pick up to count frame indices separated by at least min_gap_frames."""
    values = [int(x) for x in list(frames)]
    rng.shuffle(values)
    chosen: list[int] = []
    for frame in values:
        if all(abs(frame - prev) >= min_gap_frames for prev in chosen):
            chosen.append(frame)
            if len(chosen) >= count:
                break
    return sorted(chosen)


def create_stereo_pair(p1_path: str, p2_path: str) -> tuple[np.ndarray, int]:
    a, sr_a = read_mono(p1_path)
    b, sr_b = read_mono(p2_path)
    if sr_a != sr_b:
        raise ValueError(f"Sample-rate mismatch: {p1_path} ({sr_a}) vs {p2_path} ({sr_b})")
    a, b = pad_to_same_length(a, b)
    stereo = np.stack([a, b], axis=1)
    return stereo, sr_a


def create_stereo_source(row: dict[str, Any], *, use_input_audio_as_source: bool) -> tuple[np.ndarray, int]:
    if use_input_audio_as_source:
        audio_path = row.get("audio_path") or row.get("natural_wav_path")
        if not audio_path:
            raise ValueError("--use-input-audio-as-source requires rows with audio_path or natural_wav_path")
        return read_stereo(audio_path)
    return create_stereo_pair(row["participant1_relpath_abs"], row["participant2_relpath_abs"])


def normalize_stereo_audio(
    stereo: np.ndarray,
    *,
    target_rms_dbfs: float = -20.0,
    peak_dbfs: float = -1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = stereo.astype(np.float32, copy=True)
    if x.size == 0:
        return x, {"enabled": False, "reason": "empty_audio"}
    rms = float(np.sqrt(np.maximum(np.mean(x ** 2), 1e-12)))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    target_rms = float(10.0 ** (target_rms_dbfs / 20.0))
    peak_ceiling = float(10.0 ** (peak_dbfs / 20.0))
    gain = target_rms / max(rms, 1e-8)
    if peak > 0.0 and peak * gain > peak_ceiling:
        gain = peak_ceiling / peak
    y = np.clip(x * gain, -peak_ceiling, peak_ceiling).astype(np.float32)
    return y, {
        "enabled": True,
        "target_rms_dbfs": float(target_rms_dbfs),
        "peak_dbfs": float(peak_dbfs),
        "gain": round(float(gain), 6),
        "gain_db": round(float(20.0 * math.log10(max(gain, 1e-12))), 3),
        "input_rms_dbfs": round(float(20.0 * math.log10(max(rms, 1e-12))), 3),
        "input_peak_dbfs": round(float(20.0 * math.log10(max(peak, 1e-12))), 3),
    }


def save_dualturn_example(
    out_root: Path,
    split: str,
    stem: str,
    stereo: np.ndarray,
    sr: int,
    meta_extra: dict[str, Any] | None = None,
    *,
    normalize_audio: bool = False,
    target_rms_dbfs: float = -20.0,
    peak_dbfs: float = -1.0,
) -> tuple[Path, Path]:
    wav_dir = out_root / split / "wav"
    json_dir = out_root / split / "json"
    wav_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    wav_path = wav_dir / f"{stem}.wav"
    json_path = json_dir / f"{stem}.json"

    audio_normalization = {"enabled": False}
    if normalize_audio:
        stereo, audio_normalization = normalize_stereo_audio(
            stereo, target_rms_dbfs=target_rms_dbfs, peak_dbfs=peak_dbfs
        )

    # Keep float WAV to avoid implicit int16 quantization / perceived loudness change.
    sf.write(str(wav_path), stereo.astype(np.float32), sr, subtype="FLOAT")

    meta = {
        "session_id": stem,
        "duration": float(len(stereo) / sr),
        "language": "en",
        "session_type": "dyadic",
        "redacted_segments": [],
        "audio_normalization": audio_normalization,
    }
    if meta_extra:
        meta.update(meta_extra)
    write_json(json_path, meta)
    return wav_path, json_path


def save_reference_natural_example(
    out_root: Path,
    split: str,
    stem: str,
    original_stereo: np.ndarray,
    sr: int,
    crop_meta: dict[str, Any],
    meta_extra: dict[str, Any] | None = None,
    *,
    normalize_audio: bool = False,
    target_rms_dbfs: float = -20.0,
    peak_dbfs: float = -1.0,
) -> tuple[Path, Path] | None:
    if not crop_meta.get("enabled"):
        return None
    start_ms = crop_meta.get("source_crop_start_ms")
    end_ms = crop_meta.get("source_crop_end_ms")
    if not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float)):
        return None
    s = max(0, min(len(original_stereo), int(round(float(start_ms) * sr / 1000.0))))
    e = max(s, min(len(original_stereo), int(round(float(end_ms) * sr / 1000.0))))
    if e <= s:
        return None

    wav_dir = out_root / split / "natural_wav"
    json_dir = out_root / split / "natural_json"
    wav_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / f"{stem}__natural.wav"
    json_path = json_dir / f"{stem}__natural.json"
    ref = original_stereo[s:e].copy()
    audio_normalization = {"enabled": False}
    if normalize_audio:
        ref, audio_normalization = normalize_stereo_audio(
            ref, target_rms_dbfs=target_rms_dbfs, peak_dbfs=peak_dbfs
        )
    sf.write(str(wav_path), ref.astype(np.float32), sr, subtype="FLOAT")
    meta = {
        "session_id": f"{stem}__natural",
        "duration": float(len(ref) / sr),
        "language": "en",
        "session_type": "dyadic",
        "source_crop_start_ms": round(float(start_ms), 1),
        "source_crop_end_ms": round(float(end_ms), 1),
        "paired_unnatural_stem": stem,
        "audio_normalization": audio_normalization,
    }
    if meta_extra:
        meta.update(meta_extra)
    write_json(json_path, meta)
    return wav_path, json_path


def manifest_row(stem: str, wav_path: Path, json_path: Path, split: str) -> dict[str, Any]:
    return {
        "id": stem,
        "session_id": stem,
        "source_type": "file",
        "audio_path": str(wav_path.resolve()),
        "json_path": str(json_path.resolve()),
        "tar_path": "",
        "member_flac": "",
        "member_json": "",
        "duration_sec": sf.info(str(wav_path)).duration,
        "language": "en",
        "session_type": "dyadic",
        "split": split,
    }


ABSOLUTE_EDIT_MS_KEYS = {
    "anchor_ms",
    "speaker_start_ms", "speaker_end_ms",
    "responder_original_start_ms", "responder_original_end_ms", "responder_new_start_ms", "responder_new_end_ms",
    "speaker_insert_ms", "responder_insert_ms",
    "cut_start_ms", "cut_end_ms",
    "backchannel_start_ms", "backchannel_end_ms",
    "insert_ms", "insert_end_ms",
    "removed_responder_start_ms", "removed_responder_end_ms",
    "next_original_speaker_start_ms", "next_original_speaker_end_ms",
    "next_turn_start_ms", "next_turn_end_ms",
    "source_start_ms", "source_end_ms",
    "inserted_backchannel_start_ms", "inserted_backchannel_end_ms",
    "hold_start_ms", "hold_end_ms",
    "next_hold_start_ms", "next_hold_end_ms",
    "next_hold_original_start_ms", "next_hold_original_end_ms",
    "next_hold_new_start_ms", "next_hold_new_end_ms",
}


def edit_anchor_ms(edit: dict[str, Any]) -> float:
    edit_type = str(edit.get("edit_type") or "")
    preferred_by_type = {
        "early_entry": ("responder_new_start_ms", "cut_start_ms", "anchor_ms"),
        "late_response": ("anchor_ms", "responder_new_start_ms"),
        "hold_instead_of_shift": ("removed_responder_start_ms", "anchor_ms"),
        "shift_instead_of_hold": ("insert_ms", "hold_end_ms"),
        "missed_backchannel": ("backchannel_start_ms", "anchor_ms"),
        "excessive_backchannel": ("insert_ms", "anchor_ms"),
    }
    keys = preferred_by_type.get(edit_type, (
        "anchor_ms", "insert_ms", "removed_responder_start_ms",
        "backchannel_start_ms", "responder_original_start_ms", "hold_end_ms",
    ))
    for key in keys:
        val = edit.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
    return 0.0


def _silent_boundary_candidates(
    stereo: np.ndarray,
    sr: int,
    threshold: float,
    lo: int,
    hi: int,
    *,
    kind: str,
    min_silence_ms: float = 350.0,
    step_ms: float = 80.0,
) -> list[int]:
    lo = max(0, min(len(stereo), lo))
    hi = max(lo, min(len(stereo), hi))
    min_sil = max(1, int(round(sr * min_silence_ms / 1000.0)))
    step = max(1, int(round(sr * step_ms / 1000.0)))
    out: list[int] = []
    if kind == "start":
        last = max(lo, hi - min_sil)
        for pos in range(lo, last + 1, step):
            if stereo_region_silent(stereo, pos, pos + min_sil, sr, threshold):
                out.append(pos)
    else:
        first = lo + min_sil
        for pos in range(first, hi + 1, step):
            if stereo_region_silent(stereo, pos - min_sil, pos, sr, threshold):
                out.append(pos)
    return out


RESPONSE_CROP_POST_MS = 500.0


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _response_duration_ms(edit: dict[str, Any]) -> float | None:
    start = _finite_float(edit.get("responder_original_start_ms"))
    end = _finite_float(edit.get("responder_original_end_ms"))
    if start is None or end is None or end <= start:
        return None
    return end - start


def _edited_timeline_required_end_ms(edit: dict[str, Any], post_ms: float = RESPONSE_CROP_POST_MS) -> float | None:
    """Right-edge guard in the edited timeline to avoid cropping the manipulated response."""
    typ = str(edit.get("edit_type") or "")
    if typ == "late_response":
        original_end = _finite_float(edit.get("responder_original_end_ms"))
        delay = _finite_float(edit.get("delay_ms"))
        if original_end is not None and delay is not None:
            return original_end + delay + post_ms
    if typ in {"early_entry", "early_interruption"}:
        new_start = _finite_float(edit.get("responder_new_start_ms"))
        dur = _response_duration_ms(edit)
        if new_start is not None and dur is not None:
            return new_start + dur + post_ms
    if typ == "shift_instead_of_hold":
        new_end = _finite_float(edit.get("next_hold_new_end_ms", edit.get("next_hold_end_ms")))
        if new_end is not None:
            return new_end + post_ms
    return None


def _source_timeline_required_end_ms(edit: dict[str, Any], post_ms: float = RESPONSE_CROP_POST_MS) -> float | None:
    """Right-edge guard in the original/natural timeline for paired natural references."""
    typ = str(edit.get("edit_type") or "")
    if typ in {"late_response", "early_entry", "early_interruption"}:
        original_end = _finite_float(edit.get("responder_original_end_ms"))
        if original_end is not None:
            return original_end + post_ms
    if typ == "shift_instead_of_hold":
        original_end = _finite_float(edit.get("next_hold_original_end_ms", edit.get("next_hold_end_ms")))
        if original_end is not None:
            return original_end + post_ms
    return None




def _edit_scorer_boundary_ms(edit: dict[str, Any]) -> list[float]:
    typ = str(edit.get("edit_type") or "")
    vals: list[float] = []

    def add(key: str) -> None:
        value = _finite_float(edit.get(key))
        if value is not None:
            vals.append(value)

    def add_list(list_key: str, *keys: str) -> None:
        items = edit.get(list_key)
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in keys:
                value = _finite_float(item.get(key))
                if value is not None:
                    vals.append(value)

    if typ == "late_response":
        add("speaker_start_ms")
        add("speaker_end_ms")
        add("responder_new_start_ms")
        original_end = _finite_float(edit.get("responder_original_end_ms"))
        delay = _finite_float(edit.get("delay_ms"))
        if original_end is not None and delay is not None:
            vals.append(original_end + delay)
        else:
            add("responder_original_end_ms")
        add("next_original_speaker_start_ms")
        add("next_original_speaker_end_ms")
    elif typ in {"early_entry", "early_interruption", "interruption"}:
        add("speaker_start_ms")
        add("speaker_end_ms")
        add("responder_new_start_ms")
        if _finite_float(edit.get("responder_new_end_ms")) is not None:
            add("responder_new_end_ms")
        else:
            new_start = _finite_float(edit.get("responder_new_start_ms"))
            dur = _response_duration_ms(edit)
            if new_start is not None and dur is not None:
                vals.append(new_start + dur)
            else:
                add("responder_original_end_ms")
        add("next_turn_start_ms")
        add("next_turn_end_ms")
    elif typ == "hold_instead_of_shift":
        for key in ("speaker_start_ms", "speaker_end_ms", "removed_responder_start_ms", "removed_responder_end_ms", "next_original_speaker_start_ms", "next_original_speaker_end_ms"):
            add(key)
    elif typ == "shift_instead_of_hold":
        for key in ("hold_start_ms", "hold_end_ms", "insert_ms", "insert_end_ms", "next_hold_start_ms", "next_hold_end_ms"):
            add(key)
    elif typ in {"missed_backchannel", "excessive_backchannel"}:
        for key in (
            "speaker_start_ms", "speaker_end_ms",
            "backchannel_start_ms", "backchannel_end_ms",
            "insert_ms", "inserted_backchannel_start_ms", "inserted_backchannel_end_ms",
            "next_turn_start_ms", "next_turn_end_ms",
        ):
            add(key)
        add_list("removed_backchannels", "start_ms", "end_ms", "speaker_start_ms", "speaker_end_ms")
        add_list("inserted_backchannels", "start_ms", "end_ms")
    else:
        add("anchor_ms")
    return [x for x in vals if math.isfinite(float(x))]


def _scorer_coverage_bounds_ms(
    edits: list[dict[str, Any]],
    *,
    context_s: float,
    future_s: float,
    unit_pre_s: float,
    unit_post_s: float = 0.0,
) -> tuple[float | None, float | None, list[float]]:
    boundaries: list[float] = []
    for edit in edits:
        boundaries.extend(_edit_scorer_boundary_ms(edit))
    boundaries = [float(x) for x in boundaries if math.isfinite(float(x))]
    if not boundaries:
        return None, None, []
    left_ms = min(boundaries) - (max(0.0, context_s) + max(0.0, unit_pre_s)) * 1000.0
    right_ms = max(boundaries) + (max(0.0, future_s) + max(0.0, unit_post_s)) * 1000.0
    return left_ms, right_ms, boundaries




def validate_protected_unit_coverage(
    edit: dict[str, Any],
    *,
    duration_ms: float,
    context_s: float,
    future_s: float,
    unit_pre_s: float,
    unit_post_s: float = 0.0,
) -> None:
    left_ms, right_ms, boundaries = _scorer_coverage_bounds_ms(
        [edit],
        context_s=context_s,
        future_s=future_s,
        unit_pre_s=unit_pre_s,
        unit_post_s=unit_post_s,
    )
    if left_ms is None or right_ms is None or not boundaries:
        return
    if left_ms < 0.0:
        raise RuntimeError(
            "protected_unit_left_context_unavailable:"
            f" first_boundary_ms={round(min(boundaries), 1)}, "
            f"required_start_ms={round(left_ms, 1)}, available_start_ms=0.0"
        )
    if right_ms > duration_ms:
        raise RuntimeError(
            "protected_unit_right_future_unavailable:"
            f" last_boundary_ms={round(max(boundaries), 1)}, "
            f"required_end_ms={round(right_ms, 1)}, available_end_ms={round(duration_ms, 1)}"
        )


def _source_crop_bounds_from_edits(crop_start_ms: float, crop_end_ms: float, edits: list[dict[str, Any]]) -> tuple[float, float]:
    def inv_late(t: float, anchor: float, delay: float) -> float:
        if t <= anchor:
            return t
        if t <= anchor + delay:
            return anchor
        return t - delay

    def inv_early(t: float, cut_start: float, global_shift: float) -> float:
        if t <= cut_start:
            return t
        return t + global_shift

    start = crop_start_ms
    end = crop_end_ms
    for edit in edits:
        typ = str(edit.get("edit_type") or "")
        if typ == "late_response":
            anchor = float(edit.get("anchor_ms", 0.0))
            delay = float(edit.get("delay_ms", 0.0))
            start = inv_late(start, anchor, delay)
            end = inv_late(end, anchor, delay)
        elif typ == "early_entry":
            cut_start = float(edit.get("cut_start_ms", 0.0))
            global_shift = float(edit.get("global_timeline_shift_ms", edit.get("advance_ms", 0.0)))
            start = inv_early(start, cut_start, global_shift)
            end = inv_early(end, cut_start, global_shift)
        elif typ == "shift_instead_of_hold":
            insert = float(edit.get("insert_ms", 0.0))
            dur = float(edit.get("insert_duration_ms", 0.0))
            start = inv_late(start, insert, dur)
            end = inv_late(end, insert, dur)
    required_source_end = max(
        [x for x in (_source_timeline_required_end_ms(e) for e in edits) if x is not None],
        default=None,
    )
    if required_source_end is not None:
        end = max(end, required_source_end)
    return round(max(0.0, start), 1), round(max(0.0, end), 1)


def crop_stereo_context(
    stereo: np.ndarray,
    sr: int,
    edits: list[dict[str, Any]],
    rng: random.Random,
    *,
    min_context_s: float,
    max_context_s: float,
    threshold: float,
    late_response_local_context_s: float | None = None,
    late_response_pre_s: float = 4.0,
    late_response_post_s: float = 1.0,
    scorer_context_s: float = 3.0,
    scorer_future_s: float = 2.0,
    scorer_unit_pre_s: float = 2.0,
    scorer_unit_post_s: float = 0.0,
    enforce_scorer_coverage: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(stereo) == 0:
        return stereo, {"enabled": False, "reason": "empty_audio"}
    total_s = len(stereo) / float(sr)
    if total_s <= max_context_s:
        return stereo, {"enabled": False, "reason": "already_short", "duration_s": round(total_s, 3)}

    min_len = int(round(min_context_s * sr))
    max_len = int(round(max_context_s * sr))
    effective_min_len = min_len
    late_response_focus_applied = False
    late_response_edits = [e for e in edits if str(e.get("edit_type") or "") == "late_response"]
    early_entry_focus_cap: int | None = None
    early_entry_edits = [e for e in edits if str(e.get("edit_type") or "") == "early_entry"]
    if early_entry_edits:
        # Keep early-entry clips focused on the manipulated response; otherwise
        # every later utterance is globally time-shifted and can sound like a
        # second event when comparing against the natural reference.
        effective_min_len = min(min_len, int(round(10.0 * sr)))
        caps = []
        for edit in early_entry_edits:
            val = edit.get("responder_new_start_ms")
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                caps.append(int(round((float(val) / 1000.0 + 3.5) * sr)))
        if caps:
            early_entry_focus_cap = max(0, min(caps))
    anchors_s = [min(max(edit_anchor_ms(e) / 1000.0, 0.0), total_s) for e in edits]
    if not anchors_s:
        anchors_s = [total_s / 2.0]
    anchor_samples = sorted(int(round(x * sr)) for x in anchors_s)
    first_anchor = anchor_samples[0]
    last_anchor = anchor_samples[-1]
    required_edited_ends = [
        int(round((x / 1000.0) * sr))
        for x in (_edited_timeline_required_end_ms(e) for e in edits)
        if x is not None
    ]
    required_end = min(len(stereo), max([last_anchor, *required_edited_ends]))
    scorer_required_start: int | None = None
    scorer_required_end: int | None = None
    scorer_boundary_ms: list[float] = []
    scorer_coverage_left_clipped = False
    scorer_coverage_right_clipped = False
    scorer_left_ms, scorer_right_ms, scorer_boundary_ms = _scorer_coverage_bounds_ms(
        edits,
        context_s=scorer_context_s,
        future_s=scorer_future_s,
        unit_pre_s=scorer_unit_pre_s,
        unit_post_s=scorer_unit_post_s,
    )
    if enforce_scorer_coverage and scorer_left_ms is not None and scorer_right_ms is not None:
        total_ms = len(stereo) * 1000.0 / float(sr)
        scorer_coverage_left_clipped = scorer_left_ms < 0.0
        scorer_coverage_right_clipped = scorer_right_ms > total_ms
        scorer_required_start = max(0, int(math.floor((scorer_left_ms / 1000.0) * sr)))
        scorer_required_end = min(len(stereo), int(math.ceil((scorer_right_ms / 1000.0) * sr)))
        required_end = max(required_end, scorer_required_end)
    anchor_span = max(0, required_end - first_anchor)

    if late_response_edits and len(late_response_edits) == len(edits) and late_response_local_context_s:
        # Keep late-response clips local while still forcing the delayed response
        # to be fully included. This fixes response truncation without inflating
        # every strong late-response sample to a 20-25s context.
        local_len = int(round(max(1.0, float(late_response_local_context_s)) * sr))
        response_safe_len = anchor_span + int(round((late_response_pre_s + late_response_post_s) * sr))
        local_max_len = max(local_len, response_safe_len)
        effective_min_len = min(effective_min_len, local_len)
        max_len = min(max_len, local_max_len)
        late_response_focus_applied = True

    if anchor_span > max_len - int(round(2.0 * sr)):
        raise RuntimeError("Edits plus response/scorer guard are too far apart for the requested short-context duration")
    if enforce_scorer_coverage and scorer_required_start is not None and scorer_required_end is not None:
        if scorer_required_end - scorer_required_start > max_len:
            raise RuntimeError("Scorer coverage guard is too wide for the requested short-context duration")

    min_side = int(round(2.0 * sr))
    target_len = max(effective_min_len, min(max_len, anchor_span + int(round(rng.uniform(18.0, 30.0) * sr))))
    target_frac = rng.uniform(0.25, 0.75)
    anchor_mid = (first_anchor + required_end) // 2

    start_lo = max(0, required_end - max_len)
    start_hi = min(first_anchor - min_side, len(stereo) - effective_min_len)
    end_lo = max(required_end, last_anchor + min_side, effective_min_len)
    end_hi = min(len(stereo), first_anchor + max_len)
    if enforce_scorer_coverage and scorer_required_start is not None and scorer_required_end is not None:
        start_hi = min(start_hi, scorer_required_start)
        end_lo = max(end_lo, scorer_required_end)
    early_entry_focus_applied = False
    if early_entry_focus_cap is not None and early_entry_focus_cap >= end_lo:
        end_hi = min(end_hi, early_entry_focus_cap)
        early_entry_focus_applied = True
    if start_hi < start_lo or end_hi < end_lo:
        raise RuntimeError("No feasible short-context bounds around all edits")

    start_cands = _silent_boundary_candidates(stereo, sr, threshold, start_lo, start_hi, kind="start")
    end_cands = _silent_boundary_candidates(stereo, sr, threshold, end_lo, end_hi, kind="end")

    best: tuple[float, int, int] | None = None
    for st in start_cands:
        for en in end_cands:
            dur = en - st
            if dur < effective_min_len or dur > max_len:
                continue
            if not (st < first_anchor and required_end < en):
                continue
            if enforce_scorer_coverage and scorer_required_start is not None and scorer_required_end is not None:
                if st > scorer_required_start or en < scorer_required_end:
                    continue
            frac = (anchor_mid - st) / float(dur)
            if frac < 0.18 or frac > 0.82:
                continue
            score = abs(dur - target_len) / sr + abs(frac - target_frac) * 1.5
            if best is None or score < best[0]:
                best = (score, st, en)
    if best is None:
        raise RuntimeError("No short-context crop with mutual-silence start/end found")

    _, start, end = best
    anchor_s = anchor_mid / float(sr)
    crop_start_ms = round(start * 1000.0 / sr, 1)
    crop_end_ms = round(end * 1000.0 / sr, 1)
    source_crop_start_ms, source_crop_end_ms = _source_crop_bounds_from_edits(crop_start_ms, crop_end_ms, edits)

    for edit in edits:
        for key in list(ABSOLUTE_EDIT_MS_KEYS):
            val = edit.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                edit[f"source_timeline_{key}"] = round(float(val), 1)
                edit[key] = round(float(val) - crop_start_ms, 1)
        nested_local_time_keys = {"start_ms", "end_ms", "speaker_start_ms", "speaker_end_ms"}
        for list_key in ("removed_backchannels", "inserted_backchannels"):
            items = edit.get(list_key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                for key in nested_local_time_keys:
                    val = item.get(key)
                    if isinstance(val, (int, float)) and math.isfinite(float(val)):
                        item[f"source_timeline_{key}"] = round(float(val), 1)
                        item[key] = round(float(val) - crop_start_ms, 1)

    return stereo[start:end].copy(), {
        "enabled": True,
        "crop_start_ms": crop_start_ms,
        "crop_end_ms": crop_end_ms,
        "source_crop_start_ms": source_crop_start_ms,
        "source_crop_end_ms": source_crop_end_ms,
        "required_edited_end_ms": round(required_end * 1000.0 / sr, 1),
        "response_crop_post_ms": RESPONSE_CROP_POST_MS,
        "scorer_coverage_enabled": bool(enforce_scorer_coverage),
        "scorer_context_s": float(scorer_context_s),
        "scorer_future_s": float(scorer_future_s),
        "scorer_unit_pre_s": float(scorer_unit_pre_s),
        "scorer_unit_post_s": float(scorer_unit_post_s),
        "scorer_required_start_ms": round(scorer_required_start * 1000.0 / sr, 1) if scorer_required_start is not None else None,
        "scorer_required_end_ms": round(scorer_required_end * 1000.0 / sr, 1) if scorer_required_end is not None else None,
        "scorer_requested_start_ms": round(float(scorer_left_ms), 1) if scorer_left_ms is not None else None,
        "scorer_requested_end_ms": round(float(scorer_right_ms), 1) if scorer_right_ms is not None else None,
        "scorer_coverage_left_clipped": bool(scorer_coverage_left_clipped),
        "scorer_coverage_right_clipped": bool(scorer_coverage_right_clipped),
        "scorer_boundary_ms": [round(float(x), 1) for x in scorer_boundary_ms],
        "duration_s": round((end - start) / float(sr), 3),
        "effective_min_context_s": round(effective_min_len / float(sr), 3),
        "early_entry_focus_crop": bool(early_entry_focus_applied),
        "early_entry_focus_post_s": 3.5 if early_entry_focus_applied else None,
        "late_response_focus_crop": bool(late_response_focus_applied),
        "late_response_local_context_s": float(late_response_local_context_s) if late_response_focus_applied else None,
        "late_response_pre_s": float(late_response_pre_s) if late_response_focus_applied else None,
        "late_response_post_s": float(late_response_post_s) if late_response_focus_applied else None,
        "anchor_source_timeline_ms": round(anchor_s * 1000.0, 1),
        "anchor_cropped_ms": round(anchor_s * 1000.0 - crop_start_ms, 1),
        "boundary_rule": "mutual_silence_start_and_end",
    }


def make_late_response(a: np.ndarray, b: np.ndarray, sr: int, rng: random.Random, frame_ms: float, threshold: float,
                       min_delay_ms: int, max_delay_ms: int, p1_path: str | Path | None = None,
                       p2_path: str | Path | None = None, turn_source: str = "metadata") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    candidate_source = "metadata_vad" if turn_source == "metadata" else turn_source
    sa, sb, va, vb, turn_sources = get_candidate_context(
        a, b, sr, frame_ms, threshold, p1_path=p1_path, p2_path=p2_path, turn_source=candidate_source
    )
    shifts = find_active_alone_shift_points_bidir(
        sa,
        sb,
        va,
        vb,
        frame_ms=frame_ms,
        pre_offset_ms=1000.0,
        post_onset_ms=1000.0,
        max_overlap_ms=200.0,
        max_gap_ms=None,
        min_pre_duration_ms=1000.0,
        min_post_duration_ms=1000.0,
    )
    if not shifts:
        raise RuntimeError("No active-alone A-B transition found for late_response")
    speaker_ch, speaker_seg, responder_seg = rng.choice(shifts)
    next_pool = sa if speaker_ch == "a" else sb
    next_same_speaker = next(
        (seg for seg in next_pool if seg.start > responder_seg.end and is_substantive_turn(seg)),
        None,
    )
    delay = int(round(sr * rng.uniform(min_delay_ms, max_delay_ms) / 1000.0))
    speaker_insert = segment_end_sample(speaker_seg, sr, frame_len)
    responder_insert = segment_start_sample(responder_seg, sr, frame_len)
    # Allow original transitions to be short-gap or mildly overlapping. Insert
    # equal-duration fill independently: after the yielding speaker's offset in
    # that channel, and before the responder's onset in the responder channel.
    # This preserves both local turns without assuming a clean shared gap.
    anchor = min(speaker_insert, responder_insert)
    splice_fade_ms = 60.0
    tone_len = delay + 2 * _fade_len_samples(sr, splice_fade_ms)
    tone = make_per_channel_response_room_tone(
        a,
        b,
        anchor,
        responder_insert,
        tone_len,
        sr,
        threshold,
        rng,
        speaker_ch=speaker_ch,
        va=va,
        vb=vb,
        frame_ms=frame_ms,
    )
    if tone is None:
        tone_a, tone_b, tone_source = make_room_tone_pair(
            a,
            b,
            anchor,
            tone_len,
            sr,
            threshold,
            rng,
            va=va,
            vb=vb,
            frame_ms=frame_ms,
        )
    else:
        tone_a, tone_b, tone_source = tone
    responder_channel = "b" if speaker_ch == "a" else "a"
    if speaker_ch == "a":
        a = _insert_with_crossfade(a, speaker_insert, delay, tone_a, sr=sr, fade_ms=splice_fade_ms)
        b = _insert_with_crossfade(b, responder_insert, delay, tone_b, sr=sr, fade_ms=splice_fade_ms)
    else:
        b = _insert_with_crossfade(b, speaker_insert, delay, tone_b, sr=sr, fade_ms=splice_fade_ms)
        a = _insert_with_crossfade(a, responder_insert, delay, tone_a, sr=sr, fade_ms=splice_fade_ms)
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = a[:n].astype(np.float32, copy=False)
        b = b[:n].astype(np.float32, copy=False)
    meta = {
        "edit_type": "late_response",
        "synthesis": "insert_equal_fill_per_channel_at_speaker_end_and_responder_start",
        "late_candidate_policy": "active_alone_ab_min1s_max200ms_overlap_clean_gap_no_aba_required",
        "insert_fill": tone_source,
        "delay_ms": round(delay * 1000.0 / sr, 1),
        "pre_response_gap_ms": round((responder_insert - speaker_insert) * 1000.0 / sr, 1),
        "max_allowed_overlap_ms": 200.0,
        "min_speaker_duration_ms": 1000.0,
        "min_responder_duration_ms": 1000.0,
        "speaker_duration_ms": segment_duration_ms(speaker_seg, frame_ms),
        "responder_duration_ms": segment_duration_ms(responder_seg, frame_ms),
        "speaker_insert_ms": sample_to_ms(speaker_insert, sr),
        "responder_insert_ms": sample_to_ms(responder_insert, sr),
        "anchor_frame": int(round(anchor / frame_len)),
        "anchor_ms": sample_to_ms(anchor, sr),
        "speaker_channel": speaker_ch,
        "responder_channel": responder_channel,
        "turn_source_a": turn_sources.get("a", ""),
        "turn_source_b": turn_sources.get("b", ""),
        "speaker_start_ms": segment_start_ms(speaker_seg, frame_ms),
        "speaker_end_ms": segment_end_ms(speaker_seg, frame_ms),
        "responder_original_start_ms": segment_start_ms(responder_seg, frame_ms),
        "responder_original_end_ms": segment_end_ms(responder_seg, frame_ms),
        "responder_new_start_ms": sample_to_ms(responder_insert + delay, sr),
        "responder_new_end_ms": sample_to_ms(segment_end_sample(responder_seg, sr, frame_len) + delay, sr),
        "speaker_text": speaker_seg.text,
        "responder_text": responder_seg.text,
    }
    if next_same_speaker is not None:
        meta.update({
            "next_original_speaker_start_ms": sample_to_ms(segment_start_sample(next_same_speaker, sr, frame_len) + delay, sr),
            "next_original_speaker_end_ms": sample_to_ms(segment_end_sample(next_same_speaker, sr, frame_len) + delay, sr),
            "next_original_speaker_text": next_same_speaker.text,
        })
    return a, b, meta


def make_early_interruption(a: np.ndarray, b: np.ndarray, sr: int, rng: random.Random, frame_ms: float, threshold: float,
                            min_advance_ms: int, max_advance_ms: int, p1_path: str | Path | None = None,
                            p2_path: str | Path | None = None, turn_source: str = "metadata") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    candidate_source = "metadata_vad" if turn_source == "metadata" else turn_source
    sa, sb, va, vb, turn_sources = get_candidate_context(
        a, b, sr, frame_ms, threshold, p1_path=p1_path, p2_path=p2_path, turn_source=candidate_source
    )
    guard = int(round(sr * 0.05))
    min_advance_samples = int(round(sr * min_advance_ms / 1000.0))
    max_advance_samples = int(round(sr * max_advance_ms / 1000.0))
    min_gap_samples = int(round(sr * 0.20))
    max_gap_samples = int(round(sr * 1.00))
    safe_shifts = []
    for cand in find_proper_shift_points_bidir(sa, sb, va, vb, frame_ms=frame_ms):
        speaker_ch, speaker_seg, responder_seg = cand
        speaker_start = segment_start_sample(speaker_seg, sr, frame_len)
        gap_s = segment_end_sample(speaker_seg, sr, frame_len)
        gap_e = segment_start_sample(responder_seg, sr, frame_len)
        gap_len = gap_e - gap_s
        if gap_len < min_gap_samples or gap_len > max_gap_samples:
            continue
        removable = max(0, gap_e - gap_s - 2 * guard)
        speaker_segments = sa if speaker_ch == "a" else sb
        next_speaker_candidates = [
            seg for seg in speaker_segments
            if seg.start > responder_seg.end and is_substantive_turn(seg)
        ]
        if not next_speaker_candidates:
            continue
        next_speaker_seg = min(next_speaker_candidates, key=lambda seg: seg.start)
        next_speaker_start = segment_start_sample(next_speaker_seg, sr, frame_len)
        speaker_quiet_budget = max(0, next_speaker_start - guard - (gap_s + guard))
        max_allowed_advance = min(
            max_advance_samples,
            max(0, gap_e - speaker_start - int(round(0.30 * sr))),
            speaker_quiet_budget,
        )
        if max_allowed_advance < min_advance_samples:
            continue
        if pair_samples_silent(a, b, gap_s, gap_e, sr, threshold):
            safe_shifts.append((
                cand, gap_s, gap_e, removable, gap_len, max_allowed_advance,
                speaker_ch, next_speaker_seg,
            ))
    if not safe_shifts:
        raise RuntimeError(
            "No synchronized A-B early_entry transition with a 0.2-1.0s silent gap, "
            "valid advance, and enough same-speaker post-offset silence"
        )

    (
        (speaker_ch, speaker_seg, responder_seg), gap_s, gap_e, removable,
        gap_len, max_allowed_advance, next_turn_ch, next_turn_seg,
    ) = rng.choice(safe_shifts)
    responder_start = segment_start_sample(responder_seg, sr, frame_len)
    if max_allowed_advance < min_advance_samples:
        raise RuntimeError("Selected transition cannot support requested early_entry advance range")
    advance = rng.randint(min_advance_samples, max_allowed_advance)
    responder_channel = "b" if speaker_ch == "a" else "a"
    overlap = 0
    extra_channel_advance = 0
    speaker_channel_cut_start = None
    speaker_channel_cut_end = None
    responder_channel_cut_start = None
    responder_channel_cut_end = None
    early_entry_kind = "non_interrupt" if advance <= removable else "interrupt"

    if advance <= removable:
        global_timeline_shift = advance
        cut_end = gap_e - guard
        cut_start = cut_end - advance
        if cut_start < gap_s + guard:
            cut_start = gap_s + guard
            cut_end = cut_start + advance
        if cut_end > gap_e - guard:
            raise RuntimeError("Safe early_entry deletion would touch responder speech")
        a, b = shift_pair_earlier_smooth(a, b, cut_start, advance, sr)
        speaker_channel_cut_start = responder_channel_cut_start = cut_start
        speaker_channel_cut_end = responder_channel_cut_end = cut_end
        responder_new_start = responder_start - advance
        synthesis = "remove_gap_shift_both_channels_earlier"
    else:
        global_timeline_shift = advance
        extra_channel_advance = advance - removable
        speaker_channel_cut_start = gap_s + guard
        speaker_channel_cut_end = speaker_channel_cut_start + advance
        responder_channel_cut_end = responder_start - guard
        responder_channel_cut_start = responder_channel_cut_end - advance
        speaker_sig = a if speaker_ch == "a" else b
        responder_sig = b if responder_channel == "b" else a
        speaker_vad = va if speaker_ch == "a" else vb
        responder_vad = vb if responder_channel == "b" else va
        silence_threshold = max(0.006, float(threshold) * 0.70)

        responder_is_safe = (
            responder_channel_cut_start >= 0
            and vad_samples_silent(
                responder_vad, responder_channel_cut_start, responder_channel_cut_end,
                sr, frame_ms, guard_ms=0.0,
            )
            and channel_samples_silent(
                responder_sig, responder_channel_cut_start, responder_channel_cut_end,
                sr, threshold,
            )
            and _max_window_rms(
                responder_sig, responder_channel_cut_start, responder_channel_cut_end,
                max(1, int(round(sr * 0.03))),
            ) <= silence_threshold
        )
        if not responder_is_safe:
            raise RuntimeError("Responder channel is not quiet enough for synchronized early_entry advance")

        speaker_is_safe = (
            speaker_channel_cut_end <= len(speaker_sig)
            and vad_samples_silent(
                speaker_vad, speaker_channel_cut_start, speaker_channel_cut_end,
                sr, frame_ms, guard_ms=0.0,
            )
            and channel_samples_silent(
                speaker_sig, speaker_channel_cut_start, speaker_channel_cut_end,
                sr, threshold,
            )
            and _max_window_rms(
                speaker_sig, speaker_channel_cut_start, speaker_channel_cut_end,
                max(1, int(round(sr * 0.03))),
            ) <= silence_threshold
        )
        if not speaker_is_safe:
            raise RuntimeError("Speaker channel has no safe post-offset silence for synchronized early_entry advance")

        tone_pack = make_room_tone_pair(
            a, b, gap_e, advance, sr, threshold, rng,
            va=va, vb=vb, frame_ms=frame_ms,
        )
        if tone_pack is None:
            tone_a = np.zeros(advance, dtype=np.float32)
            tone_b = np.zeros(advance, dtype=np.float32)
        else:
            tone_a, tone_b, _ = tone_pack

        if speaker_ch == "a":
            a = shift_channel_earlier_with_fill(
                a, speaker_channel_cut_start, speaker_channel_cut_end, tone_a
            )
            b = shift_channel_earlier_with_fill(
                b, responder_channel_cut_start, responder_channel_cut_end, tone_b
            )
        else:
            b = shift_channel_earlier_with_fill(
                b, speaker_channel_cut_start, speaker_channel_cut_end, tone_b
            )
            a = shift_channel_earlier_with_fill(
                a, responder_channel_cut_start, responder_channel_cut_end, tone_a
            )
        for sig, position in (
            (a if speaker_ch == "a" else b, speaker_channel_cut_start),
            (b if responder_channel == "b" else a, responder_channel_cut_start),
        ):
            apply_edge_fades(
                sig,
                max(0, position - int(round(0.02 * sr))),
                min(len(sig), position + int(round(0.02 * sr))),
                sr,
            )
        if len(a) != len(b):
            raise RuntimeError("Synchronized early_entry cuts produced unequal channel lengths")
        cut_start = responder_channel_cut_start
        cut_end = responder_channel_cut_end
        responder_new_start = responder_start - advance
        overlap = max(0, gap_s - responder_new_start)
        synthesis = "equal_channel_local_silence_cuts_shift_b_and_future_a_earlier"

    return a, b, {
        "edit_type": "early_entry",
        "legacy_edit_type": "early_interruption",
        "early_entry_kind": early_entry_kind,
        "synthesis": synthesis,
        "advance_ms": round(advance * 1000.0 / sr, 1),
        "advance_ratio_of_removable_gap": round(advance / max(float(removable), 1.0), 3),
        "global_timeline_shift_ms": round(global_timeline_shift * 1000.0 / sr, 1),
        "extra_channel_advance_ms": round(extra_channel_advance * 1000.0 / sr, 1),
        "overlap_ms": round(overlap * 1000.0 / sr, 1),
        "secondary_shift_qc": {
            "risk": "none",
            "reason": "both channel tails advance by the same duration using channel-local silence cuts",
            "mitigation": "speaker cut ends before the next same-speaker utterance; responder cut ends before B onset",
        },
        "gap_ms": round(gap_len * 1000.0 / sr, 1),
        "removable_gap_ms": round(removable * 1000.0 / sr, 1),
        "cut_start_ms": sample_to_ms(cut_start, sr),
        "cut_end_ms": sample_to_ms(cut_end, sr),
        "speaker_channel_cut_start_ms": sample_to_ms(speaker_channel_cut_start, sr),
        "speaker_channel_cut_end_ms": sample_to_ms(speaker_channel_cut_end, sr),
        "responder_channel_cut_start_ms": sample_to_ms(responder_channel_cut_start, sr),
        "responder_channel_cut_end_ms": sample_to_ms(responder_channel_cut_end, sr),
        "anchor_frame": int(responder_seg.start),
        "anchor_ms": segment_start_ms(responder_seg, frame_ms),
        "speaker_channel": speaker_ch,
        "responder_channel": responder_channel,
        "turn_source_a": turn_sources.get("a", ""),
        "turn_source_b": turn_sources.get("b", ""),
        "speaker_start_ms": segment_start_ms(speaker_seg, frame_ms),
        "speaker_end_ms": segment_end_ms(speaker_seg, frame_ms),
        "responder_original_start_ms": segment_start_ms(responder_seg, frame_ms),
        "responder_original_end_ms": segment_end_ms(responder_seg, frame_ms),
        "responder_new_start_ms": round(responder_new_start * 1000.0 / sr, 1),
        "responder_new_end_ms": round(
            (responder_new_start + (
                segment_end_sample(responder_seg, sr, frame_len)
                - segment_start_sample(responder_seg, sr, frame_len)
            )) * 1000.0 / sr,
            1,
        ),
        "next_turn_channel": next_turn_ch,
        "next_turn_start_ms": round(
            (segment_start_sample(next_turn_seg, sr, frame_len) - global_timeline_shift)
            * 1000.0 / sr,
            1,
        ),
        "next_turn_end_ms": round(
            (segment_end_sample(next_turn_seg, sr, frame_len) - global_timeline_shift)
            * 1000.0 / sr,
            1,
        ),
        "speaker_text": speaker_seg.text,
        "responder_text": responder_seg.text,
        "next_turn_text": next_turn_seg.text,
    }


def make_interruption(a: np.ndarray, b: np.ndarray, sr: int, rng: random.Random, frame_ms: float, threshold: float,
                      min_advance_ms: int, max_advance_ms: int, p1_path: str | Path | None = None,
                      p2_path: str | Path | None = None, turn_source: str = "metadata") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    candidate_source = "metadata_vad" if turn_source == "metadata" else turn_source
    sa, sb, va, vb, turn_sources = get_candidate_context(
        a, b, sr, frame_ms, threshold, p1_path=p1_path, p2_path=p2_path, turn_source=candidate_source
    )
    min_overlap = int(round(sr * min_advance_ms / 1000.0))
    max_overlap = int(round(sr * max_advance_ms / 1000.0))
    candidates = []
    for cand in find_proper_shift_points_bidir(sa, sb, va, vb, frame_ms=frame_ms):
        speaker_ch, speaker_seg, responder_seg = cand
        speaker_end = segment_end_sample(speaker_seg, sr, frame_len)
        responder_start = segment_start_sample(responder_seg, sr, frame_len)
        speaker_start = segment_start_sample(speaker_seg, sr, frame_len)
        # Responder channel should be quiet during the region we delete from it.
        max_allowed = min(max_overlap, max(0, speaker_end - speaker_start - int(round(0.3 * sr))))
        if max_allowed < min_overlap:
            continue
        new_start = speaker_end - max_allowed
        responder_sig = b if speaker_ch == "a" else a
        if _max_window_rms(
            responder_sig, new_start, responder_start, max(1, int(round(sr * 0.03)))
        ) <= max(0.006, float(threshold) * 0.70):
            candidates.append((cand, speaker_start, speaker_end, responder_start, max_allowed))
    if not candidates:
        raise RuntimeError("No candidate shift with enough responder-side silence for interruption")

    (speaker_ch, speaker_seg, responder_seg), speaker_start, speaker_end, responder_start, max_allowed = rng.choice(candidates)
    overlap = int(round(rng.uniform(min_overlap, max_allowed)))
    new_start = speaker_end - overlap
    cut_start = new_start
    cut_end = responder_start
    cut_len = cut_end - cut_start
    if cut_len <= 0:
        raise RuntimeError("Invalid interruption shift length")
    tone_a, tone_b, tone_source = make_room_tone_pair(a, b, cut_end, cut_len, sr, threshold, rng, va=va, vb=vb, frame_ms=frame_ms)

    if speaker_ch == "a":
        b = shift_channel_earlier_with_fill(b, cut_start, cut_end, tone_b)
        apply_edge_fades(b, max(0, cut_start - int(round(0.02 * sr))), min(len(b), cut_start + int(round(0.02 * sr))), sr)
        responder_channel = "b"
    else:
        a = shift_channel_earlier_with_fill(a, cut_start, cut_end, tone_a)
        apply_edge_fades(a, max(0, cut_start - int(round(0.02 * sr))), min(len(a), cut_start + int(round(0.02 * sr))), sr)
        responder_channel = "a"

    return a, b, {
        "edit_type": "interruption",
        "synthesis": "shift_responder_channel_earlier_overlap_speaker",
        "insert_fill": tone_source,
        "overlap_ms": sample_to_ms(overlap, sr),
        "advance_ms": sample_to_ms(cut_len, sr),
        "cut_start_ms": sample_to_ms(cut_start, sr),
        "cut_end_ms": sample_to_ms(cut_end, sr),
        "anchor_ms": sample_to_ms(new_start, sr),
        "speaker_channel": speaker_ch,
        "responder_channel": responder_channel,
        "turn_source_a": turn_sources.get("a", ""),
        "turn_source_b": turn_sources.get("b", ""),
        "speaker_start_ms": segment_start_ms(speaker_seg, frame_ms),
        "speaker_end_ms": segment_end_ms(speaker_seg, frame_ms),
        "responder_original_start_ms": segment_start_ms(responder_seg, frame_ms),
        "responder_original_end_ms": segment_end_ms(responder_seg, frame_ms),
        "responder_new_start_ms": sample_to_ms(new_start, sr),
        "responder_new_end_ms": sample_to_ms(new_start + (segment_end_sample(responder_seg, sr, frame_len) - segment_start_sample(responder_seg, sr, frame_len)), sr),
        "speaker_text": speaker_seg.text,
        "responder_text": responder_seg.text,
    }




def best_overlapping_segment(segs: list[Segment], start: int, end: int) -> Segment | None:
    best: tuple[int, Segment] | None = None
    for seg in segs:
        overlap = max(0, min(seg.end + 1, end) - max(seg.start, start))
        if overlap <= 0:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, seg)
    return best[1] if best is not None else None


def first_segment_after(segs: list[Segment], frame: int) -> Segment | None:
    cands = [seg for seg in segs if seg.start > frame and is_substantive_turn(seg)]
    return min(cands, key=lambda seg: seg.start) if cands else None


def add_context_turn_metadata(
    meta: dict[str, Any],
    *,
    current: Segment | None,
    next_turn: Segment | None,
    frame_ms: float,
    prefix: str = "speaker",
) -> None:
    if current is not None:
        meta[f"{prefix}_start_ms"] = segment_start_ms(current, frame_ms)
        meta[f"{prefix}_end_ms"] = segment_end_ms(current, frame_ms)
        meta[f"{prefix}_text"] = current.text
    if next_turn is not None:
        meta["next_turn_start_ms"] = segment_start_ms(next_turn, frame_ms)
        meta["next_turn_end_ms"] = segment_end_ms(next_turn, frame_ms)
        meta["next_turn_text"] = next_turn.text


def make_missed_backchannel(a: np.ndarray, b: np.ndarray, sr: int, rng: random.Random, frame_ms: float, threshold: float,
                              p1_path: str | Path | None = None, p2_path: str | Path | None = None,
                              turn_source: str = "silero", bc_remove_count: int = 1,
                              min_missed_backchannels: int = 1) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    sa, sb, va, vb, turn_sources = get_candidate_context(
        a, b, sr, frame_ms, threshold, p1_path=p1_path, p2_path=p2_path,
        turn_source=turn_source, min_transcript_ms=40.0
    )
    candidates: list[tuple[str, Segment]] = []
    candidates.extend(("b", seg) for seg in find_isolated_backchannels(sb, vb, va, frame_ms=frame_ms))
    candidates.extend(("a", seg) for seg in find_isolated_backchannels(sa, va, vb, frame_ms=frame_ms))
    candidates = sorted(candidates, key=lambda x: (x[1].start, x[0]))
    min_needed = max(1, int(min_missed_backchannels))
    if len(candidates) < min_needed:
        raise RuntimeError(
            f"Only {len(candidates)} candidate backchannels found for missed_backchannel; "
            f"need at least {min_needed}"
        )

    remove_count = max(1, min(int(bc_remove_count), len(candidates)))
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected = sorted(shuffled[:remove_count], key=lambda x: (x[1].start, x[0]))

    removed: list[dict[str, Any]] = []
    for channel, seg in selected:
        s = segment_start_sample(seg, sr, frame_len)
        e = segment_end_sample(seg, sr, frame_len)
        if channel == "b":
            zero_region_smooth(b, s, e, sr)
            speaker_segs = sa
        else:
            zero_region_smooth(a, s, e, sr)
            speaker_segs = sb
        speaker_seg_i = best_overlapping_segment(speaker_segs, seg.start, seg.end + 1)
        removed.append({
            "channel": channel,
            "start_ms": segment_start_ms(seg, frame_ms),
            "end_ms": segment_end_ms(seg, frame_ms),
            "text": seg.text,
            "speaker_start_ms": segment_start_ms(speaker_seg_i, frame_ms) if speaker_seg_i is not None else None,
            "speaker_end_ms": segment_end_ms(speaker_seg_i, frame_ms) if speaker_seg_i is not None else None,
            "speaker_text": speaker_seg_i.text if speaker_seg_i is not None else "",
        })

    first_ch, first_seg = selected[0]
    first_speaker_segs = sa if first_ch == "b" else sb
    speaker_seg = best_overlapping_segment(first_speaker_segs, first_seg.start, first_seg.end + 1)
    last_end = max(seg.end for _, seg in selected)
    next_turn = first_segment_after(sa + sb, last_end)
    meta = {
        "edit_type": "missed_backchannel",
        "channel": first_ch,
        "anchor_frame": int(first_seg.start),
        "anchor_ms": segment_start_ms(first_seg, frame_ms),
        "turn_source_a": turn_sources.get("a", ""),
        "turn_source_b": turn_sources.get("b", ""),
        "backchannel_start_ms": segment_start_ms(first_seg, frame_ms),
        "backchannel_end_ms": segment_end_ms(first_seg, frame_ms),
        "backchannel_text": first_seg.text,
        "removed_backchannels": removed,
        "removed_backchannel_count": int(len(removed)),
        "requested_removed_backchannel_count": int(bc_remove_count),
        "candidate_backchannel_count": int(len(candidates)),
        "min_missed_backchannels": int(min_needed),
    }
    add_context_turn_metadata(meta, current=speaker_seg, next_turn=next_turn, frame_ms=frame_ms)
    return a, b, meta


def make_excessive_backchannel(a: np.ndarray, b: np.ndarray, sr: int, rng: random.Random, frame_ms: float, threshold: float,
                               bc_insert_gain: float = 1.0, p1_path: str | Path | None = None,
                               p2_path: str | Path | None = None, turn_source: str = "silero",
                               bc_insert_count: int = 2, bc_insert_min_gap_ms: float = 800.0,
                               allow_bc_source_reuse: bool = False,
                               max_bc_source_reuse_per_clip: int = 0,
                               allow_empty_bc_text: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    sa, sb, va, vb, turn_sources = get_candidate_context(
        a, b, sr, frame_ms, threshold, p1_path=p1_path, p2_path=p2_path,
        turn_source=turn_source, min_transcript_ms=40.0
    )
    bcands_b = find_isolated_backchannels(sb, vb, va, frame_ms=frame_ms)
    bcands_a = find_isolated_backchannels(sa, va, vb, frame_ms=frame_ms)
    source_pool: list[dict[str, Any]] = []
    for channel, sig, segs in (("b", b, bcands_b), ("a", a, bcands_a)):
        for seg in segs:
            clip_start, clip_end, clip_text, clip_text_source = backchannel_clip_bounds(
                sig, seg, sr=sr, frame_len=frame_len
            )
            clip = copy_clip(sig, clip_start, clip_end)
            if clip.size == 0:
                continue
            if not allow_empty_bc_text and not (str(clip_text).strip() or str(seg.text).strip()):
                continue
            source_pool.append({
                "source_id": int(len(source_pool)),
                "channel": channel,
                "seg": seg,
                "clip": clip,
                "clip_start": int(clip_start),
                "clip_end": int(clip_end),
                "clip_text": clip_text,
                "clip_text_source": clip_text_source,
                "clip_frames": max(1, int(math.ceil(len(clip) / float(frame_len)))),
            })
    if not source_pool:
        raise RuntimeError("No source backchannel found for excessive_backchannel")

    desired = max(1, int(bc_insert_count))
    max_reuse_per_clip = max(0, int(max_bc_source_reuse_per_clip))
    max_uses_per_clip = 1 + max_reuse_per_clip
    if not allow_bc_source_reuse:
        max_uses_per_clip = 1
        max_reuse_per_clip = 0
    by_channel = {
        "a": [item for item in source_pool if str(item["channel"]) == "a"],
        "b": [item for item in source_pool if str(item["channel"]) == "b"],
    }
    channel_order = [ch for ch in ("a", "b") if by_channel[ch]]
    rng.shuffle(channel_order)

    chosen_frames: list[int] = []
    speaker_seg: Segment | None = None
    selected_pool: list[dict[str, Any]] = []
    source_channel = ""
    target_sig: np.ndarray | None = None
    last_candidate_error = "No target speaking region for excessive_backchannel"

    for candidate_channel in channel_order:
        pool = list(by_channel[candidate_channel])
        rng.shuffle(pool)
        if len(pool) * max_uses_per_clip < desired:
            last_candidate_error = (
                f"Only {len(pool)} textual source backchannels found for channel {candidate_channel}; "
                f"need capacity for {desired} insertions with max reuse {max_reuse_per_clip} per clip."
            )
            continue
        if candidate_channel == "b":
            candidate_target_sig = b
            target_speaker_vad = va
            target_listener_vad = vb
            target_speaker_segs = sa
        else:
            candidate_target_sig = a
            target_speaker_vad = vb
            target_listener_vad = va
            target_speaker_segs = sb

        max_clip_frames = max(int(item["clip_frames"]) for item in pool)
        min_gap_frames = max(max_clip_frames, int(math.ceil(float(bc_insert_min_gap_ms) / frame_ms)))

        # Keep multiple excessive backchannels as one local event in the same
        # listener channel. A-source BC clips are only pasted into A; B-source
        # BC clips are only pasted into B.
        target_windows = [
            seg for seg in target_speaker_segs
            if seg.end > seg.start and (seg.end - seg.start + 1) >= max_clip_frames
        ]
        rng.shuffle(target_windows)
        for window in target_windows:
            lo = max(0, int(window.start))
            hi = min(len(target_speaker_vad), int(window.end) + 1 - max_clip_frames)
            if hi <= lo:
                continue
            active = [int(lo + x) for x in np.flatnonzero(target_speaker_vad[lo:hi] > 0)]
            silent_listener = [
                int(f) for f in active
                if _silent(target_listener_vad, int(f), int(f) + max_clip_frames)
            ]
            if not silent_listener:
                last_candidate_error = (
                    f"No listener-silent insertion frames found for channel {candidate_channel} "
                    f"inside target turn {frame_to_ms(lo, frame_ms):.1f}-{frame_to_ms(hi, frame_ms):.1f}ms."
                )
                continue
            candidates = silent_listener
            trial = pick_spaced_frames(candidates, count=desired, min_gap_frames=min_gap_frames, rng=rng)
            if len(trial) >= desired:
                chosen_frames = trial[:desired]
                speaker_seg = window
                selected_pool = pool
                source_channel = candidate_channel
                target_sig = candidate_target_sig
                break
            last_candidate_error = (
                f"Only {len(trial)} local insertion frames found for channel {candidate_channel}; "
                f"need {desired}. Try smaller --bc-insert-count or --bc-insert-min-gap-ms."
            )
        if chosen_frames:
            break

    if len(chosen_frames) < desired or target_sig is None or not selected_pool:
        raise RuntimeError(last_candidate_error)
    source_pool = selected_pool

    source_sequence: list[dict[str, Any]] = []
    for _ in range(max_uses_per_clip):
        round_sources = list(source_pool)
        rng.shuffle(round_sources)
        if (
            source_sequence
            and len(round_sources) > 1
            and int(round_sources[0]["source_id"]) == int(source_sequence[-1]["source_id"])
        ):
            round_sources = round_sources[1:] + round_sources[:1]
        source_sequence.extend(round_sources)
    source_sequence = source_sequence[:desired]

    inserted: list[dict[str, Any]] = []
    source_use_counts: dict[int, int] = {}
    for idx, start_frame in enumerate(chosen_frames):
        source = source_sequence[idx]
        source_id = int(source["source_id"])
        source_reuse_index = int(source_use_counts.get(source_id, 0))
        source_use_counts[source_id] = source_reuse_index + 1
        clip = np.asarray(source["clip"], dtype=np.float32)
        clip_start = int(source["clip_start"])
        clip_end = int(source["clip_end"])
        clip_frames = int(source["clip_frames"])
        paste_add(target_sig, clip, frame_to_sample(start_frame, frame_len), gain=bc_insert_gain)
        inserted_end_frame = start_frame + clip_frames
        inserted.append({
            "start_ms": frame_to_ms(int(start_frame), frame_ms),
            "end_ms": round(frame_to_ms(int(start_frame), frame_ms) + sample_to_ms(clip_end - clip_start, sr), 1),
            "insert_frame": int(start_frame),
            "source_channel": str(source["channel"]),
            "source_backchannel_start_ms": sample_to_ms(clip_start, sr),
            "source_backchannel_end_ms": sample_to_ms(clip_end, sr),
            "backchannel_duration_ms": sample_to_ms(clip_end - clip_start, sr),
            "backchannel_text": str(source["clip_text"]),
            "backchannel_segment_text": str(source["seg"].text),
            "backchannel_text_source": str(source["clip_text_source"]),
            "source_id": int(source_id),
            "source_reused": bool(source_reuse_index > 0),
            "source_reuse_index": int(source_reuse_index),
        })

    first_start = int(chosen_frames[0])
    last_end = max(int(frame) + int(source_sequence[idx]["clip_frames"]) for idx, frame in enumerate(chosen_frames))
    if speaker_seg is None:
        speaker_seg = best_overlapping_segment(target_speaker_segs, first_start, last_end)
    next_turn = first_segment_after(sa + sb, last_end)
    first = inserted[0]
    meta = {
        "edit_type": "excessive_backchannel",
        "source_channel": source_channel,
        "insert_frame": int(first_start),
        "insert_ms": float(first["start_ms"]),
        "bc_insert_gain": float(bc_insert_gain),
        "bc_insert_count": int(len(inserted)),
        "requested_bc_insert_count": int(desired),
        "bc_insert_min_gap_ms": float(bc_insert_min_gap_ms),
        "source_backchannel_candidate_count": int(len(source_pool)),
        "source_backchannel_reuse_count": int(sum(1 for item in inserted if item.get("source_reused"))),
        "allow_bc_source_reuse": bool(allow_bc_source_reuse),
        "max_bc_source_reuse_per_clip": int(max_reuse_per_clip),
        "allow_empty_bc_text": bool(allow_empty_bc_text),
        "turn_source_a": turn_sources.get("a", ""),
        "turn_source_b": turn_sources.get("b", ""),
        "source_backchannel_start_ms": float(first["source_backchannel_start_ms"]),
        "source_backchannel_end_ms": float(first["source_backchannel_end_ms"]),
        "inserted_backchannel_start_ms": float(first["start_ms"]),
        "inserted_backchannel_end_ms": float(first["end_ms"]),
        "backchannel_duration_ms": float(first["backchannel_duration_ms"]),
        "backchannel_text": str(first["backchannel_text"]),
        "backchannel_segment_text": str(first["backchannel_segment_text"]),
        "backchannel_text_source": str(first["backchannel_text_source"]),
        "inserted_backchannels": inserted,
    }
    add_context_turn_metadata(meta, current=speaker_seg, next_turn=next_turn, frame_ms=frame_ms)
    return a, b, meta


def make_hold_instead_of_shift(a: np.ndarray, b: np.ndarray, sr: int, rng: random.Random, frame_ms: float, threshold: float,
                               hold_extension_ms: int = 800, p1_path: str | Path | None = None,
                               p2_path: str | Path | None = None, turn_source: str = "silero",
                               hold_shift_remove_pad_ms: float = 500.0,
                               hold_shift_max_gap_ms: float | None = None,
                               hold_shift_min_edited_hold_gap_ms: float = 100.0,
                               hold_shift_max_edited_hold_gap_ms: float | None = None,
                               hold_shift_min_responder_s: float = 1.0,
                               hold_shift_max_responder_s: float = 4.0,
                               hold_shift_require_return_proper: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    candidate_source = "metadata_vad" if turn_source == "metadata" else turn_source
    sa, sb, va, vb, turn_sources = get_candidate_context(
        a, b, sr, frame_ms, threshold, p1_path=p1_path, p2_path=p2_path, turn_source=candidate_source
    )
    if hold_shift_require_return_proper:
        shifts = find_shift_then_original_speaker_returns(
            sa,
            sb,
            va,
            vb,
            frame_ms=frame_ms,
            max_gap_ms=hold_shift_max_gap_ms,
            require_return_proper=True,
        )
    else:
        shifts = find_active_alone_shift_then_original_speaker_returns(
            sa,
            sb,
            va,
            vb,
            frame_ms=frame_ms,
            max_gap_ms=hold_shift_max_gap_ms,
            max_overlap_ms=0.0,
            min_mutual_silence_ms=100.0,
        )
    # Require clear textual turns; otherwise the generated metadata is not auditable.
    shifts = [
        x for x in shifts
        if _has_text(x[1]) and _has_text(x[2]) and _has_text(x[3]) and _distinct_text(x[1], x[2])
        and not _is_question_like(x[2])
        and float(hold_shift_min_responder_s) <= (segment_end_ms(x[2], frame_ms) - segment_start_ms(x[2], frame_ms)) / 1000.0 <= float(hold_shift_max_responder_s)
        and float(hold_shift_min_edited_hold_gap_ms) <= (segment_start_ms(x[3], frame_ms) - segment_end_ms(x[1], frame_ms))
        and (
            hold_shift_max_edited_hold_gap_ms is None
            or (segment_start_ms(x[3], frame_ms) - segment_end_ms(x[1], frame_ms)) <= float(hold_shift_max_edited_hold_gap_ms)
        )
    ]
    if not shifts:
        raise RuntimeError("No textual non-question shift-return found for hold_instead_of_shift")
    speaker_ch, speaker_seg, responder_seg, next_speaker_seg = rng.choice(shifts)
    resp_s = segment_start_sample(responder_seg, sr, frame_len)
    resp_e = segment_end_sample(responder_seg, sr, frame_len)
    pad_samples = max(0, int(round(sr * float(hold_shift_remove_pad_ms) / 1000.0)))
    prev_end = segment_end_sample(speaker_seg, sr, frame_len)
    next_start = segment_start_sample(next_speaker_seg, sr, frame_len)
    # For hold-instead-of-shift, the responder should be absent for the whole
    # edited hold region. Replacing the entire previous-A-end -> next-A-start
    # span avoids VAD-truncated responder tails leaking past resp_e.
    remove_s = prev_end
    remove_e = next_start
    if remove_e <= remove_s:
        remove_s = max(prev_end, resp_s - pad_samples)
        remove_e = min(next_start, resp_e + pad_samples)
    if remove_e <= remove_s:
        remove_s, remove_e = resp_s, resp_e
    resp_len = max(0, remove_e - remove_s)
    tone = make_per_channel_response_room_tone(
        a,
        b,
        segment_end_sample(speaker_seg, sr, frame_len),
        resp_s,
        resp_len,
        sr,
        threshold,
        rng,
        speaker_ch=speaker_ch,
        va=va,
        vb=vb,
        frame_ms=frame_ms,
    )
    if tone is None:
        tone_a, tone_b, tone_source = make_room_tone_pair(
            a,
            b,
            remove_s,
            resp_len,
            sr,
            threshold,
            rng,
            va=va,
            vb=vb,
            frame_ms=frame_ms,
        )
    else:
        tone_a, tone_b, tone_source = tone
    if speaker_ch == "a":
        replace_region_with_fill_smooth(b, remove_s, remove_e, tone_b, sr)
        removed_channel = "b"
        held_channel = "a"
    else:
        replace_region_with_fill_smooth(a, remove_s, remove_e, tone_a, sr)
        removed_channel = "a"
        held_channel = "b"
    return a, b, {
        "edit_type": "hold_instead_of_shift",
        "synthesis": "remove_responder_replace_with_room_tone",
        "removed_fill": tone_source,
        "question_removed_filtered": True,
        "proper_return_required": bool(hold_shift_require_return_proper),
        "proper_transition_definition": (
            "strict_proper_shift_with_vap_silence_window"
            if hold_shift_require_return_proper
            else "active_alone_shift_pre_offset_1000ms_post_onset_1000ms_min_mutual_silence_100ms_no_overlap_no_vap_eval_silence_window"
        ),
        "hold_shift_max_gap_ms": float(hold_shift_max_gap_ms) if hold_shift_max_gap_ms is not None else None,
        "hold_shift_min_edited_hold_gap_ms": float(hold_shift_min_edited_hold_gap_ms),
        "hold_shift_max_edited_hold_gap_ms": float(hold_shift_max_edited_hold_gap_ms) if hold_shift_max_edited_hold_gap_ms is not None else None,
        "hold_shift_min_responder_s": float(hold_shift_min_responder_s),
        "hold_shift_max_responder_s": float(hold_shift_max_responder_s),
        "removed_responder_duration_ms": segment_end_ms(responder_seg, frame_ms) - segment_start_ms(responder_seg, frame_ms),
        "edited_hold_gap_ms": segment_start_ms(next_speaker_seg, frame_ms) - segment_end_ms(speaker_seg, frame_ms),
        "hold_shift_remove_pad_ms": float(hold_shift_remove_pad_ms),
        "removed_region_policy": "full_hold_gap_prev_end_to_next_start",
        "removed_region_start_ms": sample_to_ms(remove_s, sr),
        "removed_region_end_ms": sample_to_ms(remove_e, sr),
        "pre_shift_gap_ms": segment_start_ms(responder_seg, frame_ms) - segment_end_ms(speaker_seg, frame_ms),
        "post_shift_gap_ms": segment_start_ms(next_speaker_seg, frame_ms) - segment_end_ms(responder_seg, frame_ms),
        "removed_frame": int(responder_seg.start),
        "removed_channel": removed_channel,
        "held_channel": held_channel,
        "turn_source_a": turn_sources.get("a", ""),
        "turn_source_b": turn_sources.get("b", ""),
        "speaker_start_ms": segment_start_ms(speaker_seg, frame_ms),
        "speaker_end_ms": segment_end_ms(speaker_seg, frame_ms),
        "removed_responder_start_ms": segment_start_ms(responder_seg, frame_ms),
        "removed_responder_end_ms": segment_end_ms(responder_seg, frame_ms),
        "next_original_speaker_start_ms": segment_start_ms(next_speaker_seg, frame_ms),
        "next_original_speaker_end_ms": segment_end_ms(next_speaker_seg, frame_ms),
        "speaker_text": speaker_seg.text,
        "next_original_speaker_text": next_speaker_seg.text,
        "removed_responder_text": responder_seg.text,
    }


def make_shift_instead_of_hold(a: np.ndarray, b: np.ndarray, sr: int, rng: random.Random, frame_ms: float, threshold: float,
                               shift_insert_gain: float = 0.9, p1_path: str | Path | None = None,
                               p2_path: str | Path | None = None, turn_source: str = "silero",
                               shift_hold_pre_silence_ms: float = 1000.0,
                               shift_hold_post_silence_ms: float = 1000.0,
                               shift_hold_min_gap_ms: float = 300.0,
                               shift_hold_max_gap_ms: float = 1000.0,
                               shift_hold_min_insert_s: float = 1.0,
                               shift_hold_max_insert_s: float = 4.0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    candidate_source = "metadata_vad" if turn_source == "metadata" else turn_source
    sa, sb, va, vb, turn_sources = get_candidate_context(
        a, b, sr, frame_ms, threshold, p1_path=p1_path, p2_path=p2_path,
        turn_source=candidate_source, min_transcript_ms=40.0
    )

    diagnostics: Counter[str] = Counter()
    proper_hold_regions = find_proper_hold_regions_bidir(sa, sb, va, vb, frame_ms=frame_ms)
    diagnostics["proper_hold_regions"] = len(proper_hold_regions)
    hold_regions = []
    for x in proper_hold_regions:
        if not _has_text(x[1]):
            diagnostics["hold_current_no_text"] += 1
            continue
        if not _has_text(x[2]):
            diagnostics["hold_next_no_text"] += 1
            continue
        if not _distinct_text(x[1], x[2]):
            diagnostics["hold_text_not_distinct"] += 1
            continue
        hold_regions.append(x)
    diagnostics["textual_hold_regions"] = len(hold_regions)
    if not proper_hold_regions:
        raise RuntimeError(
            "No strict proper A-A hold region found for shift_instead_of_hold; "
            f"shift_hold_diagnostics={json.dumps(dict(diagnostics.most_common()), ensure_ascii=False)}"
        )
    if not hold_regions:
        raise RuntimeError(
            "No textual hold region found for shift_instead_of_hold; "
            f"shift_hold_diagnostics={json.dumps(dict(diagnostics.most_common()), ensure_ascii=False)}"
        )

    rng.shuffle(hold_regions)
    last_error = "unknown"
    for hold_ch, curr, nxt in hold_regions:
        inserted_ch = "b" if hold_ch == "a" else "a"
        gap_s = segment_end_sample(curr, sr, frame_len)
        gap_e = segment_start_sample(nxt, sr, frame_len)
        gap_len = gap_e - gap_s
        gap_ms = sample_to_ms(gap_len, sr)
        if gap_ms < float(shift_hold_min_gap_ms) or gap_ms > float(shift_hold_max_gap_ms):
            diagnostics["hold_gap_outside_range"] += 1
            last_error = f"hold gap {gap_ms:.1f}ms outside requested range {shift_hold_min_gap_ms}-{shift_hold_max_gap_ms}ms"
            continue
        # The 1s pre-offset/post-onset condition is enforced by
        # _proper_hold_transition(): the holder must be active-alone before
        # the first turn offset and after the next turn onset. The gap itself
        # is only the mutual-silence HOLD gap, so it should stay short.
        if not pair_samples_silent(a, b, gap_s, gap_e, sr, threshold):
            diagnostics["hold_gap_not_waveform_silent"] += 1
            last_error = "hold gap is not waveform-silent"
            continue
        candidate_segs = sa if inserted_ch == "a" else sb
        # Source turn comes from the same original conversation but outside this HOLD neighborhood.
        avoid_s = max(0, curr.start - int(round(15_000.0 / frame_ms)))
        avoid_e = nxt.end + int(round(15_000.0 / frame_ms))
        source_pool = []
        min_insert_samples = int(round(sr * float(shift_hold_min_insert_s)))
        max_insert_samples = int(round(sr * float(shift_hold_max_insert_s)))
        for seg in candidate_segs:
            diagnostics["source_candidates_checked"] += 1
            if not _has_text(seg, min_words=3):
                diagnostics["source_no_text_or_too_few_words"] += 1
                continue
            if not is_substantive_turn(seg, min_frames=10, min_words=3):
                diagnostics["source_not_substantive"] += 1
                continue
            if not (seg.end < avoid_s or seg.start > avoid_e):
                diagnostics["source_inside_avoid_window"] += 1
                continue
            src_s = segment_start_sample(seg, sr, frame_len)
            src_e = segment_end_sample(seg, sr, frame_len)
            dur = src_e - src_s
            if dur < min_insert_samples:
                diagnostics["source_too_short"] += 1
                continue
            if dur > max_insert_samples:
                diagnostics["source_too_long"] += 1
                continue
            diagnostics["source_eligible"] += 1
            source_pool.append((inserted_ch, seg, dur))
        if not source_pool:
            diagnostics["no_source_pool_after_filters"] += 1
            last_error = "No textual off-window source turn for shift_instead_of_hold"
            continue

        src_ch, src, src_dur = rng.choice(source_pool)
        clip = copy_clip(
            a if src_ch == "a" else b,
            segment_start_sample(src, sr, frame_len),
            segment_end_sample(src, sr, frame_len),
        )
        if clip.size == 0:
            diagnostics["empty_source_clip"] += 1
            last_error = "Empty source clip for shift_instead_of_hold"
            continue
        apply_edge_fades(clip, 0, len(clip), sr)

        insert_sample = gap_s + max(0, gap_len // 2)
        tone_a, tone_b, tone_source = make_room_tone_pair(
            a, b, insert_sample, src_dur, sr, threshold, rng, va=va, vb=vb, frame_ms=frame_ms
        )

        # Insert the shift turn into the HOLD gap and insert equal-duration room
        # tone in the holder channel, so both channels remain aligned after edit.
        a, b = shift_pair_later(a, b, insert_sample, src_dur, fill=(tone_a, tone_b), sr=sr, fade_ms=60.0)
        if inserted_ch == "a":
            paste_replace(a, clip, insert_sample, gain=shift_insert_gain)
            inserted_channel = "a"
        else:
            paste_replace(b, clip, insert_sample, gain=shift_insert_gain)
            inserted_channel = "b"
        insert_duration_ms = sample_to_ms(src_dur, sr)
        insert_end_sample = insert_sample + src_dur
        next_hold_original_start_ms = segment_start_ms(nxt, frame_ms)
        next_hold_original_end_ms = segment_end_ms(nxt, frame_ms)
        return a, b, {
            "edit_type": "shift_instead_of_hold",
            "synthesis": "insert_shift_turn_with_global_room_tone",
            "insert_fill": tone_source,
            "duration_preserved": False,
            "insert_frame": int(round(insert_sample / frame_len)),
            "insert_ms": sample_to_ms(insert_sample, sr),
            "insert_end_ms": sample_to_ms(insert_end_sample, sr),
            "insert_duration_ms": insert_duration_ms,
            "global_timeline_insert_ms": insert_duration_ms,
            "source_channel": src_ch,
            "inserted_channel": inserted_channel,
            "hold_channel": hold_ch,
            "shift_insert_gain": float(shift_insert_gain),
            "proper_hold_pre_offset_ms": 1000.0,
            "proper_hold_post_onset_ms": 1000.0,
            "shift_hold_pre_silence_ms": sample_to_ms(insert_sample - gap_s, sr),
            "shift_hold_post_silence_ms": sample_to_ms(gap_e - insert_sample, sr),
            "shift_hold_legacy_pre_silence_arg_ms": float(shift_hold_pre_silence_ms),
            "shift_hold_legacy_post_silence_arg_ms": float(shift_hold_post_silence_ms),
            "insert_position": "midpoint_of_original_mutual_silence_gap",
            "shift_hold_min_gap_ms": float(shift_hold_min_gap_ms),
            "shift_hold_max_gap_ms": float(shift_hold_max_gap_ms),
            "original_hold_gap_ms": float(gap_ms),
            "shift_hold_min_insert_s": float(shift_hold_min_insert_s),
            "shift_hold_max_insert_s": float(shift_hold_max_insert_s),
            "actual_pre_shift_silence_ms": sample_to_ms(insert_sample - gap_s, sr),
            "actual_post_shift_silence_ms": sample_to_ms(gap_e - insert_sample, sr),
            "turn_source_a": turn_sources.get("a", ""),
            "turn_source_b": turn_sources.get("b", ""),
            "source_start_ms": segment_start_ms(src, frame_ms),
            "source_end_ms": segment_end_ms(src, frame_ms),
            "hold_start_ms": segment_start_ms(curr, frame_ms),
            "hold_end_ms": segment_end_ms(curr, frame_ms),
            "next_hold_original_start_ms": next_hold_original_start_ms,
            "next_hold_original_end_ms": next_hold_original_end_ms,
            "next_hold_new_start_ms": next_hold_original_start_ms + insert_duration_ms,
            "next_hold_new_end_ms": next_hold_original_end_ms + insert_duration_ms,
            "next_hold_start_ms": next_hold_original_start_ms + insert_duration_ms,
            "next_hold_end_ms": next_hold_original_end_ms + insert_duration_ms,
            "source_duration_ms": insert_duration_ms,
            "inserted_text": src.text,
            "source_text": src.text,
            "hold_text": curr.text,
            "next_hold_text": nxt.text,
            "shift_hold_diagnostics": dict(diagnostics.most_common()),
        }
    raise RuntimeError(
        f"{last_error}; shift_hold_diagnostics={json.dumps(dict(diagnostics.most_common()), ensure_ascii=False)}"
    )

EDIT_FNS = {
    "late_response": make_late_response,
    "early_entry": make_early_interruption,
    "early_interruption": make_early_interruption,
    "interruption": make_interruption,
    "missed_backchannel": make_missed_backchannel,
    "excessive_backchannel": make_excessive_backchannel,
    "hold_instead_of_shift": make_hold_instead_of_shift,
    "shift_instead_of_hold": make_shift_instead_of_hold,
}


def load_natural_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    nat = df[df["naturalness"] == 1].copy()
    nat = nat[nat["participant1_relpath_abs"].notna() & nat["participant2_relpath_abs"].notna()].copy()
    nat = nat.drop_duplicates(subset=["participant1_relpath_abs", "participant2_relpath_abs"]).reset_index(drop=True)
    return nat


def sample_natural(
    csv_path: Path,
    out_root: Path,
    n_pairs: int,
    split: str,
    seed: int,
) -> None:
    nat = load_natural_rows(csv_path)
    if len(nat) < n_pairs:
        raise ValueError(f"Requested {n_pairs} natural pairs but only {len(nat)} available")
    rng = random.Random(seed)
    idxs = rng.sample(list(range(len(nat))), n_pairs)

    rows = []
    natural_records = []
    for k, idx in enumerate(idxs):
        row = nat.iloc[idx].to_dict()
        stereo, sr = create_stereo_pair(row["participant1_relpath_abs"], row["participant2_relpath_abs"])
        stem = f"natural_{k:03d}"
        wav_path, json_path = save_dualturn_example(
            out_root=out_root,
            split=split,
            stem=stem,
            stereo=stereo,
            sr=sr,
            meta_extra={
                "source_csv": str(csv_path),
                "augmentation_type": "natural",
                "participant1_abs": row["participant1_relpath_abs"],
                "participant2_abs": row["participant2_relpath_abs"],
                "participant1_id": row["participant1_id"],
                "participant2_id": row["participant2_id"],
            },
        )
        rows.append(manifest_row(stem, wav_path, json_path, split))
        row["natural_stem"] = stem
        row["natural_wav_path"] = str(wav_path.resolve())
        row["natural_json_path"] = str(json_path.resolve())
        natural_records.append(row)

    manifest_dir = out_root / "manifests"
    write_manifest(manifest_dir / f"{split}.csv", rows)
    write_manifest(manifest_dir / f"{split}_all_sessions.csv", rows)
    if not (manifest_dir / "all_sessions.csv").exists():
        write_manifest(manifest_dir / "all_sessions.csv", rows)
    pd.DataFrame(natural_records).to_csv(out_root / "natural_rows.csv", index=False)
    print(f"Saved {len(rows)} natural pairs -> {out_root}")




def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ordered_natural_sources(csv_path: Path) -> pd.DataFrame:
    """
    Supports two inputs:
      1) natural_rows.csv with participant*_relpath_abs columns
      2) natural manifest csv with audio_path/json_path columns; we recover participant abs from json
    """
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("No natural rows found")

    if "participant1_relpath_abs" in df.columns and "participant2_relpath_abs" in df.columns:
        out = df.copy()
        if "source_natural_stem" not in out.columns:
            if "natural_stem" in out.columns:
                out["source_natural_stem"] = out["natural_stem"]
            elif "session_id" in out.columns:
                out["source_natural_stem"] = out["session_id"]
            elif "id" in out.columns:
                out["source_natural_stem"] = out["id"]
            else:
                out["source_natural_stem"] = [f"natural_{i:03d}" for i in range(len(out))]
        if "participant1_id" not in out.columns:
            out["participant1_id"] = ""
        if "participant2_id" not in out.columns:
            out["participant2_id"] = ""
        return out.reset_index(drop=True)

    required = {"json_path", "audio_path"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Unsupported natural_csv format at {csv_path}. Need participant*_relpath_abs columns "
            f"or manifest columns including json_path/audio_path."
        )

    rows = []
    for _, row in df.iterrows():
        meta = _read_json(row["json_path"])
        rows.append({
            "source_natural_stem": str(meta.get("session_id") or row.get("session_id") or row.get("id") or ""),
            "participant1_relpath_abs": meta["participant1_abs"],
            "participant2_relpath_abs": meta["participant2_abs"],
            "participant1_id": meta.get("participant1_id", ""),
            "participant2_id": meta.get("participant2_id", ""),
            "audio_path": row.get("audio_path", ""),
            "json_path": row.get("json_path", ""),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def _stable_seed(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _edit_time(edit: dict[str, Any]) -> float | None:
    val = edit_anchor_ms(edit)
    return float(val) if math.isfinite(float(val)) else None


def _interval_overlap_ratio(a0: float | None, a1: float | None, b0: float | None, b1: float | None) -> float:
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (a0, a1, b0, b1)):
        return 0.0
    a0f, a1f, b0f, b1f = float(a0), float(a1), float(b0), float(b1)
    if a1f <= a0f or b1f <= b0f:
        return 0.0
    inter = max(0.0, min(a1f, b1f) - max(a0f, b0f))
    return inter / max(1e-6, min(a1f - a0f, b1f - b0f))


def edits_conflict(new_edit: dict[str, Any], prior_edits: list[dict[str, Any]], min_gap_ms: float = 8000.0) -> str | None:
    new_t = _edit_time(new_edit)
    for old in prior_edits:
        old_t = _edit_time(old)
        if new_t is not None and old_t is not None and abs(new_t - old_t) < min_gap_ms:
            return f"too_close_to_existing_event:{abs(new_t - old_t):.1f}ms"
        sp_overlap = _interval_overlap_ratio(
            new_edit.get("speaker_start_ms"), new_edit.get("speaker_end_ms"),
            old.get("speaker_start_ms"), old.get("speaker_end_ms"),
        )
        resp_overlap = _interval_overlap_ratio(
            new_edit.get("responder_original_start_ms"), new_edit.get("responder_original_end_ms"),
            old.get("responder_original_start_ms"), old.get("responder_original_end_ms"),
        )
        if sp_overlap > 0.5 or resp_overlap > 0.5:
            return "overlaps_existing_turn"
    return None


def _apply_edit_n_times(
    edit_type: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    sr: int,
    frame_ms: float,
    threshold: float,
    min_delay_ms: int,
    max_delay_ms: int,
    min_advance_ms: int,
    max_advance_ms: int,
    edits_per_sample: int,
    max_tries_per_sample: int,
    seed: int,
    min_required_edits: int = 1,
    hold_extension_ms: int = 800,
    hold_shift_remove_pad_ms: float = 100.0,
    hold_shift_max_gap_ms: float | None = None,
    hold_shift_min_edited_hold_gap_ms: float = 500.0,
    hold_shift_max_edited_hold_gap_ms: float | None = 1500.0,
    hold_shift_min_responder_s: float = 1.2,
    hold_shift_max_responder_s: float = 8.0,
    hold_shift_require_return_proper: bool = False,
    shift_insert_gain: float = 0.9,
    shift_hold_pre_silence_ms: float = 1000.0,
    shift_hold_post_silence_ms: float = 1000.0,
    shift_hold_min_gap_ms: float = 300.0,
    shift_hold_max_gap_ms: float = 1000.0,
    shift_hold_min_insert_s: float = 1.0,
    shift_hold_max_insert_s: float = 4.0,
    bc_insert_gain: float = 1.0,
    bc_insert_count: int = 2,
    bc_insert_min_gap_ms: float = 800.0,
    bc_remove_count: int = 1,
    min_missed_backchannels: int = 1,
    allow_bc_source_reuse: bool = False,
    max_bc_source_reuse_per_clip: int = 0,
    allow_empty_bc_text: bool = False,
    p1_path: str | Path | None = None,
    p2_path: str | Path | None = None,
    turn_source: str = "silero",
    short_context: bool = False,
    min_context_s: float = 20.0,
    max_context_s: float = 25.0,
    late_response_local_context_s: float | None = None,
    late_response_pre_s: float = 4.0,
    late_response_post_s: float = 1.0,
    scorer_context_s: float = 3.0,
    scorer_future_s: float = 2.0,
    scorer_unit_pre_s: float = 2.0,
    scorer_unit_post_s: float = 0.0,
    enforce_scorer_coverage: bool = True,
    require_complete_protected_units: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Soft mode:
      - try to apply up to edits_per_sample edits
      - if we cannot reach that target, keep the maximum successful number
      - only fail if successful edits < min_required_edits
    """
    import inspect

    fn = EDIT_FNS[edit_type]
    sig = inspect.signature(fn)
    kwargs = {
        "sr": sr,
        "frame_ms": frame_ms,
        "threshold": threshold,
        "min_delay_ms": min_delay_ms,
        "max_delay_ms": max_delay_ms,
        "min_advance_ms": min_advance_ms,
        "max_advance_ms": max_advance_ms,
        "hold_extension_ms": hold_extension_ms,
        "hold_shift_remove_pad_ms": hold_shift_remove_pad_ms,
        "hold_shift_max_gap_ms": hold_shift_max_gap_ms,
        "hold_shift_min_edited_hold_gap_ms": hold_shift_min_edited_hold_gap_ms,
        "hold_shift_max_edited_hold_gap_ms": hold_shift_max_edited_hold_gap_ms,
        "hold_shift_min_responder_s": hold_shift_min_responder_s,
        "hold_shift_max_responder_s": hold_shift_max_responder_s,
        "hold_shift_require_return_proper": hold_shift_require_return_proper,
        "shift_insert_gain": shift_insert_gain,
        "shift_hold_pre_silence_ms": shift_hold_pre_silence_ms,
        "shift_hold_post_silence_ms": shift_hold_post_silence_ms,
        "shift_hold_min_gap_ms": shift_hold_min_gap_ms,
        "shift_hold_max_gap_ms": shift_hold_max_gap_ms,
        "shift_hold_min_insert_s": shift_hold_min_insert_s,
        "shift_hold_max_insert_s": shift_hold_max_insert_s,
        "bc_insert_gain": bc_insert_gain,
        "bc_insert_count": bc_insert_count,
        "bc_insert_min_gap_ms": bc_insert_min_gap_ms,
        "bc_remove_count": bc_remove_count,
        "min_missed_backchannels": min_missed_backchannels,
        "allow_bc_source_reuse": allow_bc_source_reuse,
        "max_bc_source_reuse_per_clip": max_bc_source_reuse_per_clip,
        "allow_empty_bc_text": allow_empty_bc_text,
        "p1_path": p1_path,
        "p2_path": p2_path,
        "turn_source": turn_source,
    }
    filtered_base = {k: v for k, v in kwargs.items() if k in sig.parameters}

    edits: list[dict[str, Any]] = []
    local_failures: list[str] = []

    requested_target = max(1, edits_per_sample)
    target = requested_target
    length_changing_edit = edit_type in {"late_response", "early_entry", "early_interruption", "shift_instead_of_hold"}
    if length_changing_edit and requested_target > 1:
        # Multiple insert/delete edits need a batch timeline transform. Until that is
        # implemented, keep one clean timing event rather than creating misaligned edits.
        target = 1
        local_failures.append("limited_to_one_length_changing_event_per_sample")
    for edit_idx in range(target):
        success = False
        for attempt in range(max_tries_per_sample):
            try:
                rng = random.Random(_stable_seed(seed, edit_type, edit_idx, attempt))
                filtered = dict(filtered_base)
                if "rng" in sig.parameters:
                    filtered["rng"] = rng
                prev_a, prev_b = a, b
                cand_a, cand_b, single_edit = fn(a.copy(), b.copy(), **filtered)
                conflict = edits_conflict(single_edit, edits)
                if conflict is not None:
                    local_failures.append(f"rejected_candidate:{conflict}")
                    a, b = prev_a, prev_b
                    continue
                if enforce_scorer_coverage or require_complete_protected_units:
                    try:
                        validate_protected_unit_coverage(
                            single_edit,
                            duration_ms=len(cand_a) * 1000.0 / float(sr),
                            context_s=scorer_context_s,
                            future_s=scorer_future_s,
                            unit_pre_s=scorer_unit_pre_s,
                            unit_post_s=scorer_unit_post_s,
                        )
                    except Exception as qc_error:
                        local_failures.append(f"rejected_candidate:{qc_error}")
                        a, b = prev_a, prev_b
                        continue
                    single_edit["scorer_boundary_coverage_validated"] = True
                    single_edit["scorer_boundary_context_s"] = float(scorer_context_s)
                    single_edit["scorer_boundary_future_s"] = float(scorer_future_s)
                    single_edit["scorer_boundary_unit_pre_s"] = float(scorer_unit_pre_s)
                    single_edit["scorer_boundary_unit_post_s"] = float(scorer_unit_post_s)
                a, b = cand_a, cand_b
                edits.append(single_edit)
                success = True
                break
            except Exception as e:
                reason = str(e)
                local_failures.append(reason)
                if is_deterministic_no_candidate_failure(edit_type, reason):
                    break
                continue
        if not success:
            # soft stop: keep what we have so far instead of failing the whole sample
            break

    if len(edits) < max(1, min_required_edits):
        raise RuntimeError(
            f"only_{len(edits)}_successful_edits_out_of_{target}; "
            f"last_error={local_failures[-1] if local_failures else 'unknown'}"
        )

    crop_meta = {"enabled": False}
    if short_context:
        stereo, crop_meta = crop_stereo_context(
            np.stack([a, b], axis=1),
            sr,
            edits,
            random.Random(_stable_seed(seed, edit_type, "short_context")),
            min_context_s=min_context_s,
            max_context_s=max_context_s,
            threshold=threshold,
            late_response_local_context_s=late_response_local_context_s,
            late_response_pre_s=late_response_pre_s,
            late_response_post_s=late_response_post_s,
            scorer_context_s=scorer_context_s,
            scorer_future_s=scorer_future_s,
            scorer_unit_pre_s=scorer_unit_pre_s,
            scorer_unit_post_s=scorer_unit_post_s,
            enforce_scorer_coverage=enforce_scorer_coverage,
        )
        a = stereo[:, 0].copy()
        b = stereo[:, 1].copy()

    return a, b, {
        "edit_type": edit_type,
        "requested_num_edits": int(requested_target),
        "num_edits": int(len(edits)),
        "edits": edits,
        "truncated": bool(len(edits) < requested_target),
        "truncation_reason": (local_failures[-1] if len(edits) < requested_target and local_failures else ""),
        "candidate_rejections": [x for x in local_failures if str(x).startswith("rejected_candidate:")],
        "short_context": crop_meta,
    }


def _pick_sources_for_type(
    df: pd.DataFrame,
    *,
    source_rows: list[dict[str, Any]],
    edit_type: str,
    per_type: int,
    count: int,
    seed: int,
    frame_ms: float,
    threshold: float,
    min_delay_ms: int,
    max_delay_ms: int,
    min_advance_ms: int,
    max_advance_ms: int,
    max_tries_per_sample: int,
    hold_extension_ms: int,
    hold_shift_remove_pad_ms: float,
    hold_shift_max_gap_ms: float | None,
    hold_shift_min_edited_hold_gap_ms: float,
    hold_shift_max_edited_hold_gap_ms: float | None,
    hold_shift_min_responder_s: float,
    hold_shift_max_responder_s: float,
    hold_shift_require_return_proper: bool,
    shift_insert_gain: float,
    shift_hold_pre_silence_ms: float,
    shift_hold_post_silence_ms: float,
    shift_hold_min_gap_ms: float,
    shift_hold_max_gap_ms: float,
    shift_hold_min_insert_s: float,
    shift_hold_max_insert_s: float,
    bc_insert_gain: float,
    bc_insert_count: int,
    bc_insert_min_gap_ms: float,
    bc_remove_count: int,
    min_missed_backchannels: int,
    allow_bc_source_reuse: bool,
    max_bc_source_reuse_per_clip: int,
    allow_empty_bc_text: bool,
    turn_source: str,
    short_context: bool,
    min_context_s: float,
    max_context_s: float,
    late_response_local_context_s: float | None,
    late_response_pre_s: float,
    late_response_post_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Keep natural order fixed:
      rows [0:per_type) -> first edit type
      rows [per_type:2*per_type) -> second edit type
      ...

    We only verify that each chosen natural can support THIS one edit type.
    Each natural row will generate exactly one unnatural sample.
    """
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in source_rows:
        try:
            stereo, sr = create_stereo_pair(row["participant1_relpath_abs"], row["participant2_relpath_abs"])
            a = stereo[:, 0].copy()
            b = stereo[:, 1].copy()
            _apply_edit_n_times(
                edit_type,
                a,
                b,
                sr=sr,
                frame_ms=frame_ms,
                threshold=threshold,
                min_delay_ms=min_delay_ms,
                max_delay_ms=max_delay_ms,
                min_advance_ms=min_advance_ms,
                max_advance_ms=max_advance_ms,
                edits_per_sample=count,
                max_tries_per_sample=max_tries_per_sample,
                seed=_stable_seed(seed, row.get("source_natural_stem", ""), edit_type, count),
                hold_extension_ms=hold_extension_ms,
                hold_shift_remove_pad_ms=hold_shift_remove_pad_ms,
                hold_shift_max_gap_ms=hold_shift_max_gap_ms,
                hold_shift_min_edited_hold_gap_ms=hold_shift_min_edited_hold_gap_ms,
                hold_shift_max_edited_hold_gap_ms=hold_shift_max_edited_hold_gap_ms,
                hold_shift_min_responder_s=hold_shift_min_responder_s,
                hold_shift_max_responder_s=hold_shift_max_responder_s,
                hold_shift_require_return_proper=hold_shift_require_return_proper,
                shift_insert_gain=shift_insert_gain,
                shift_hold_pre_silence_ms=shift_hold_pre_silence_ms,
                shift_hold_post_silence_ms=shift_hold_post_silence_ms,
                shift_hold_min_gap_ms=shift_hold_min_gap_ms,
                shift_hold_max_gap_ms=shift_hold_max_gap_ms,
                shift_hold_min_insert_s=shift_hold_min_insert_s,
                shift_hold_max_insert_s=shift_hold_max_insert_s,
                bc_insert_gain=bc_insert_gain,
                bc_insert_count=bc_insert_count,
                bc_insert_min_gap_ms=bc_insert_min_gap_ms,
                bc_remove_count=bc_remove_count,
                min_missed_backchannels=min_missed_backchannels,
                allow_bc_source_reuse=allow_bc_source_reuse,
                max_bc_source_reuse_per_clip=max_bc_source_reuse_per_clip,
                allow_empty_bc_text=allow_empty_bc_text,
                p1_path=row["participant1_relpath_abs"],
                p2_path=row["participant2_relpath_abs"],
                turn_source=turn_source,
                short_context=short_context,
                min_context_s=min_context_s,
                max_context_s=max_context_s,
                late_response_local_context_s=late_response_local_context_s,
                late_response_pre_s=late_response_pre_s,
                late_response_post_s=late_response_post_s,
            )
            selected.append(row)
        except Exception as e:
            failures.append({
                "source_natural_stem": row.get("source_natural_stem", ""),
                "edit_type": edit_type,
                "edit_count": int(count),
                "participant1_abs": row.get("participant1_relpath_abs", ""),
                "participant2_abs": row.get("participant2_relpath_abs", ""),
                "reason": str(e),
            })
        if len(selected) >= per_type:
            break

    return selected, failures


def make_unnatural(
    natural_csv: Path,
    out_root: Path,
    split: str,
    per_type: int,
    seed: int,
    frame_ms: float,
    threshold: float,
    min_delay_ms: int,
    max_delay_ms: int,
    min_advance_ms: int,
    max_advance_ms: int,
    max_tries_per_type: int = 200,
    edits_per_sample: int = 1,
    max_tries_per_sample: int = 40,
    edit_counts: str | None = None,
    hold_extension_ms: int = 800,
    hold_shift_remove_pad_ms: float = 100.0,
    hold_shift_max_gap_ms: float | None = None,
    hold_shift_min_edited_hold_gap_ms: float = 500.0,
    hold_shift_max_edited_hold_gap_ms: float | None = 1500.0,
    hold_shift_min_responder_s: float = 1.2,
    hold_shift_max_responder_s: float = 8.0,
    hold_shift_require_return_proper: bool = False,
    shift_insert_gain: float = 0.9,
    shift_hold_pre_silence_ms: float = 1000.0,
    shift_hold_post_silence_ms: float = 1000.0,
    shift_hold_min_gap_ms: float = 300.0,
    shift_hold_max_gap_ms: float = 1000.0,
    shift_hold_min_insert_s: float = 1.0,
    shift_hold_max_insert_s: float = 4.0,
    bc_insert_gain: float = 1.0,
    bc_insert_count: int = 2,
    bc_insert_min_gap_ms: float = 800.0,
    bc_remove_count: int = 1,
    min_missed_backchannels: int = 1,
    allow_bc_source_reuse: bool = False,
    max_bc_source_reuse_per_clip: int = 0,
    allow_empty_bc_text: bool = False,
    turn_source: str = "silero",
    short_context: bool = False,
    min_context_s: float = 20.0,
    max_context_s: float = 25.0,
    late_response_local_context_s: float | None = None,
    late_response_pre_s: float = 4.0,
    late_response_post_s: float = 1.0,
    scorer_context_s: float = 3.0,
    scorer_future_s: float = 2.0,
    scorer_unit_pre_s: float = 2.0,
    scorer_unit_post_s: float = 0.0,
    enforce_scorer_coverage: bool = True,
    require_complete_protected_units: bool = False,
    normalize_audio: bool = True,
    target_rms_dbfs: float = -20.0,
    peak_dbfs: float = -1.0,
    types: list[str] | None = None,
    use_input_audio_as_source: bool = False,
    reuse_input_audio_as_natural_reference: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
) -> None:
    """
    Blockwise soft mode requested by user:

      first 10 naturals  -> late_response
      next 10 naturals   -> early_entry
      next 10 naturals   -> missed_backchannel
      next 10 naturals   -> excessive_backchannel
      next 10 naturals   -> hold_instead_of_shift
      next 10 naturals   -> shift_instead_of_hold

    Each natural source generates exactly ONE unnatural sample per requested edit-count.
    If a sample cannot reach the requested number of edits, keep the maximum successful number.
    Only skip a sample when zero edits can be applied.
    """
    edit_types = list(types or UNNATURAL_TYPES)
    unknown = [t for t in edit_types if t not in EDIT_FNS]
    if unknown:
        raise ValueError(f"Unknown edit types: {unknown}; valid={sorted(EDIT_FNS)}")

    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"--shard-index must satisfy 0 <= index < num_shards; "
            f"got index={shard_index}, num_shards={num_shards}"
        )
    if num_shards > 1 and len(edit_types) != 1:
        raise ValueError("Sharded generation currently requires exactly one --types value")

    df = load_ordered_natural_sources(natural_csv).copy()
    if len(df) == 0:
        raise ValueError("No natural rows found")
    df["_source_natural_global_index"] = np.arange(len(df), dtype=np.int64)
    if num_shards > 1:
        original_count = len(df)
        df = df.iloc[shard_index::num_shards].reset_index(drop=True)
        per_type = min(int(per_type), len(df))
        print(
            f"Shard {shard_index}/{num_shards}: assigned {len(df)}/{original_count} "
            f"source rows; per_type target={per_type}"
        )

    if edit_counts:
        counts = [int(x.strip()) for x in str(edit_counts).split(",") if x.strip()]
    else:
        counts = [int(edits_per_sample)]
    if not counts:
        counts = [int(edits_per_sample)]

    needed_sources = per_type * len(edit_types)
    if len(df) < needed_sources:
        raise ValueError(
            f"Need at least {needed_sources} natural rows for blockwise generation "
            f"({per_type} per type × {len(edit_types)} types), but only {len(df)} were provided."
        )

    ordered_rows = [r.to_dict() for _, r in df.iterrows()]
    rows = []
    records = []
    failures: list[dict[str, Any]] = []
    generation_log: list[dict[str, Any]] = []
    used_source_indices: set[int] = set()
    failure_fields = [
        "source_natural_stem",
        "source_natural_global_index",
        "edit_type",
        "requested_edit_count",
        "participant1_abs",
        "participant2_abs",
        "reason",
        "compact_reason",
    ]
    failure_live_path = out_root / f"{split}_failures_live.csv"
    failure_reason_counts: Counter[str] = Counter()

    for type_idx, edit_type in enumerate(edit_types):
        block_start = type_idx * per_type
        block_end = block_start + per_type
        candidate_rows = ordered_rows[block_start:]
        successes_for_type = 0

        failures_at_type_start = len(failures)
        type_failure_reason_counts: Counter[str] = Counter()
        pbar_desc = f"{edit_type} ok 0/{per_type} ({type_idx + 1}/{len(edit_types)})"
        with tqdm(
            candidate_rows,
            total=len(candidate_rows),
            desc=pbar_desc,
            unit="src",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for local_rank, row in enumerate(pbar):
                if successes_for_type >= per_type:
                    break
                source_index = int(row.get("_source_natural_global_index", block_start + local_rank))
                if source_index in used_source_indices:
                    continue
                source_stem = str(row.get("source_natural_stem") or f"natural_{source_index:03d}")
                pbar.set_description(f"{edit_type} ok {successes_for_type}/{per_type} ({type_idx + 1}/{len(edit_types)})")
                pbar.set_postfix(
                    ok=f"{successes_for_type}/{per_type}",
                    fail_type=len(failures) - failures_at_type_start,
                    fail_total=len(failures),
                    src=source_stem[:24],
                    refresh=False,
                )
                generation_log.append({
                    "source_natural_stem": source_stem,
                    "source_natural_global_index": int(source_index),
                    "assigned_edit_type": edit_type,
                    "requested_edit_counts": ",".join(str(c) for c in counts),
                })
                row_had_success = False

                for count in counts:
                    try:
                        stereo, sr = create_stereo_source(row, use_input_audio_as_source=use_input_audio_as_source)
                        a = stereo[:, 0].copy()
                        b = stereo[:, 1].copy()
                        a2, b2, compound_meta = _apply_edit_n_times(
                            edit_type,
                            a,
                            b,
                            sr=sr,
                            frame_ms=frame_ms,
                            threshold=threshold,
                            min_delay_ms=min_delay_ms,
                            max_delay_ms=max_delay_ms,
                            min_advance_ms=min_advance_ms,
                            max_advance_ms=max_advance_ms,
                            edits_per_sample=count,
                            max_tries_per_sample=max_tries_per_sample,
                            seed=_stable_seed(seed, source_stem, edit_type, count),
                            min_required_edits=1,
                            hold_extension_ms=hold_extension_ms,
                            hold_shift_remove_pad_ms=hold_shift_remove_pad_ms,
                            hold_shift_max_gap_ms=hold_shift_max_gap_ms,
                            hold_shift_min_edited_hold_gap_ms=hold_shift_min_edited_hold_gap_ms,
                            hold_shift_max_edited_hold_gap_ms=hold_shift_max_edited_hold_gap_ms,
                            hold_shift_min_responder_s=hold_shift_min_responder_s,
                            hold_shift_max_responder_s=hold_shift_max_responder_s,
                            hold_shift_require_return_proper=hold_shift_require_return_proper,
                            shift_insert_gain=shift_insert_gain,
                            shift_hold_pre_silence_ms=shift_hold_pre_silence_ms,
                            shift_hold_post_silence_ms=shift_hold_post_silence_ms,
                            shift_hold_min_gap_ms=shift_hold_min_gap_ms,
                            shift_hold_max_gap_ms=shift_hold_max_gap_ms,
                            shift_hold_min_insert_s=shift_hold_min_insert_s,
                            shift_hold_max_insert_s=shift_hold_max_insert_s,
                            bc_insert_gain=bc_insert_gain,
                            bc_insert_count=bc_insert_count,
                            bc_insert_min_gap_ms=bc_insert_min_gap_ms,
                            bc_remove_count=bc_remove_count,
                            min_missed_backchannels=min_missed_backchannels,
                            allow_bc_source_reuse=allow_bc_source_reuse,
                            max_bc_source_reuse_per_clip=max_bc_source_reuse_per_clip,
                            allow_empty_bc_text=allow_empty_bc_text,
                            p1_path=row["participant1_relpath_abs"],
                            p2_path=row["participant2_relpath_abs"],
                            turn_source=turn_source,
                            short_context=short_context,
                            min_context_s=min_context_s,
                            max_context_s=max_context_s,
                            late_response_local_context_s=late_response_local_context_s,
                            late_response_pre_s=late_response_pre_s,
                            late_response_post_s=late_response_post_s,
                            scorer_context_s=scorer_context_s,
                            scorer_future_s=scorer_future_s,
                            scorer_unit_pre_s=scorer_unit_pre_s,
                            scorer_unit_post_s=scorer_unit_post_s,
                            enforce_scorer_coverage=enforce_scorer_coverage,
                            require_complete_protected_units=require_complete_protected_units,
                        )
                    except Exception as e:
                        reason = str(e)
                        compact_reason = compact_failure_reason(reason)
                        failure_rec = {
                            "source_natural_stem": source_stem,
                            "source_natural_global_index": int(source_index),
                            "edit_type": edit_type,
                            "requested_edit_count": int(count),
                            "participant1_abs": row.get("participant1_relpath_abs", ""),
                            "participant2_abs": row.get("participant2_relpath_abs", ""),
                            "reason": reason,
                            "compact_reason": compact_reason,
                        }
                        failures.append(failure_rec)
                        failure_reason_counts[compact_reason] += 1
                        type_failure_reason_counts[compact_reason] += 1
                        append_csv_row(failure_live_path, failure_fields, failure_rec)
                        top_reason, top_count = type_failure_reason_counts.most_common(1)[0]
                        pbar.set_description(f"{edit_type} ok {successes_for_type}/{per_type} ({type_idx + 1}/{len(edit_types)})")
                        pbar.set_postfix(
                            ok=f"{successes_for_type}/{per_type}",
                            fail_type=len(failures) - failures_at_type_start,
                            fail_total=len(failures),
                            last_fail=compact_reason[:42],
                            top_fail=f"{top_reason[:32]}:{top_count}",
                            src=source_stem[:24],
                            refresh=True,
                        )
                        continue

                    stereo_u = np.stack([a2, b2], axis=1)
                    actual_k = int(compound_meta.get("num_edits", 0))
                    stem_suffix = ""
                    if edit_type in {"early_entry", "early_interruption"}:
                        edits_for_name = compound_meta.get("edits", [])
                        if edits_for_name:
                            kind = str(edits_for_name[0].get("early_entry_kind") or "").strip()
                            if kind in {"interrupt", "non_interrupt"}:
                                stem_suffix = f"__{kind}"
                    stem = f"{source_stem}__{edit_type}{stem_suffix}__k{actual_k}"
                    if len(counts) > 1:
                        stem = f"{source_stem}__{edit_type}{stem_suffix}__req{count}__k{actual_k}"

                    reference_meta = {}
                    if reuse_input_audio_as_natural_reference:
                        input_audio = row.get("audio_path") or row.get("natural_wav_path")
                        input_json = row.get("json_path") or row.get("natural_json_path")
                        if not input_audio:
                            raise ValueError("--reuse-input-audio-as-natural-reference requires rows with audio_path or natural_wav_path")
                        reference_meta = {
                            "natural_reference_wav": str(Path(input_audio).resolve()),
                            "natural_reference_json": str(Path(input_json).resolve()) if input_json else "",
                            "natural_reference_mode": "input_audio",
                        }
                    else:
                        ref_paths = save_reference_natural_example(
                            out_root=out_root,
                            split=split,
                            stem=stem,
                            original_stereo=stereo,
                            sr=sr,
                            crop_meta=compound_meta.get("short_context", {}),
                            meta_extra={
                                "source_csv": str(natural_csv),
                                "augmentation_type": "natural_reference",
                                "participant1_abs": row["participant1_relpath_abs"],
                                "participant2_abs": row["participant2_relpath_abs"],
                                "participant1_id": row.get("participant1_id", ""),
                                "participant2_id": row.get("participant2_id", ""),
                                "source_natural_stem": source_stem,
                                "turn_source": turn_source,
                                "vad_config": dict(SILERO_CONFIG) if turn_source == "silero" else {},
                            },
                            normalize_audio=normalize_audio,
                            target_rms_dbfs=target_rms_dbfs,
                            peak_dbfs=peak_dbfs,
                        )
                        if ref_paths is not None:
                            reference_meta = {
                                "natural_reference_wav": str(ref_paths[0].resolve()),
                                "natural_reference_json": str(ref_paths[1].resolve()),
                                "natural_reference_mode": "generated_crop",
                            }

                    wav_path, json_path = save_dualturn_example(
                        out_root=out_root,
                        split=split,
                        stem=stem,
                        stereo=stereo_u,
                        sr=sr,
                        meta_extra={
                            "source_csv": str(natural_csv),
                            "augmentation_type": edit_type,
                            "participant1_abs": row["participant1_relpath_abs"],
                            "participant2_abs": row["participant2_relpath_abs"],
                            "participant1_id": row.get("participant1_id", ""),
                            "participant2_id": row.get("participant2_id", ""),
                            "source_natural_stem": source_stem,
                            "source_natural_global_index": int(source_index),
                            "assigned_edit_type": edit_type,
                            "requested_edit_count": int(count),
                            "actual_edit_count": int(actual_k),
                            "turn_source": turn_source,
                            "source_audio_mode": "input_audio" if use_input_audio_as_source else "participant_pair_raw",
                            "vad_config": dict(SILERO_CONFIG) if turn_source == "silero" else {},
                            "edit_meta": compound_meta,
                            **reference_meta,
                        },
                        normalize_audio=normalize_audio,
                        target_rms_dbfs=target_rms_dbfs,
                        peak_dbfs=peak_dbfs,
                    )
                    rows.append(manifest_row(stem, wav_path, json_path, split))
                    rec = dict(row)
                    rec.update({
                        "source_natural_stem": source_stem,
                        "source_natural_global_index": int(source_index),
                        "assigned_edit_type": edit_type,
                        "unnatural_stem": stem,
                        "unnatural_wav_path": str(wav_path.resolve()),
                        "unnatural_json_path": str(json_path.resolve()),
                        "edit_type": edit_type,
                        "requested_edit_count": int(count),
                        "actual_edit_count": int(actual_k),
                        "num_edits": int(actual_k),
                        "truncated": bool(compound_meta.get("truncated", False)),
                        "edits_json": json.dumps(compound_meta.get("edits", []), ensure_ascii=False),
                    })
                    records.append(rec)
                    row_had_success = True

                if row_had_success:
                    successes_for_type += 1
                    used_source_indices.add(source_index)
                    pbar.set_description(f"{edit_type} ok {successes_for_type}/{per_type} ({type_idx + 1}/{len(edit_types)})")
                    pbar.set_postfix(
                        ok=f"{successes_for_type}/{per_type}",
                        fail_type=len(failures) - failures_at_type_start,
                        fail_total=len(failures),
                        rows=len(rows),
                        src=source_stem[:24],
                        refresh=True,
                    )
                    pbar.write(f"[{edit_type}] success {successes_for_type}/{per_type}: {source_stem}")

        if type_failure_reason_counts:
            top_items = "; ".join(
                f"{reason}={count}" for reason, count in type_failure_reason_counts.most_common(8)
            )
            print(f"[{edit_type}] failure summary: {top_items}")

    if failure_reason_counts:
        all_top_items = "; ".join(
            f"{reason}={count}" for reason, count in failure_reason_counts.most_common(12)
        )
        print(f"All failure summary: {all_top_items}")

    manifest_dir = out_root / "manifests"
    write_manifest(manifest_dir / f"{split}.csv", rows)
    write_manifest(manifest_dir / f"{split}_all_sessions.csv", rows)
    pd.DataFrame(records).to_csv(out_root / f"generated_{split}_rows.csv", index=False)
    pd.DataFrame(generation_log).to_csv(out_root / f"{split}_generation_log.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_root / f"{split}_failures.csv", index=False)

    total_requested = per_type * len(edit_types) * len(counts)
    print(
        f"Saved {len(rows)} blockwise-soft unnatural pairs -> {out_root} "
        f"(requested up to {total_requested}; each sample keeps the maximum successful "
        f"number of edits instead of failing if it cannot reach the target)."
    )


def segment_to_label(seg: Segment, frame_ms: float) -> dict[str, Any]:
    return {
        "start_ms": segment_start_ms(seg, frame_ms),
        "end_ms": segment_end_ms(seg, frame_ms),
        "start_s": round(float(seg.start_s), 4) if seg.start_s is not None else round(seg.start * frame_ms / 1000.0, 4),
        "end_s": round(float(seg.end_s), 4) if seg.end_s is not None else round((seg.end + 1) * frame_ms / 1000.0, 4),
        "start_frame": int(seg.start),
        "end_frame": int(seg.end),
        "duration_ms": segment_duration_ms(seg, frame_ms),
        "text": seg.text,
        "source": seg.source,
    }


def export_vad_labels(wav_path: Path, output_path: Path, frame_ms: float = 80.0) -> None:
    audio, sr = sf.read(str(wav_path), always_2d=True)
    audio = audio.astype(np.float32)
    channels: dict[str, Any] = {}
    for ch in range(audio.shape[1]):
        segs, source = load_silero_segments(
            audio[:, ch], sr, None, frame_ms=frame_ms, min_transcript_ms=40.0
        )
        channels[f"ch{ch}"] = {
            "source": source,
            "segments": [segment_to_label(seg, frame_ms) for seg in segs],
        }
    write_json(output_path, {
        "wav_path": str(wav_path.resolve()),
        "sample_rate": int(sr),
        "duration_s": round(float(len(audio)) / float(sr), 4),
        "frame_ms": float(frame_ms),
        "vad_model": "silero_vad",
        "vad_config": dict(SILERO_CONFIG),
        "channels": channels,
    })
    print(f"Wrote VAD labels -> {output_path}")


def _first_shift_hold_edit(meta: dict[str, Any]) -> dict[str, Any]:
    edit_meta = meta.get("edit_meta", {}) if isinstance(meta, dict) else {}
    edits = edit_meta.get("edits", []) if isinstance(edit_meta, dict) else []
    if not isinstance(edits, list) or not edits:
        raise ValueError("JSON does not contain edit_meta.edits[0]")
    edit = edits[0]
    if not isinstance(edit, dict):
        raise ValueError("edit_meta.edits[0] is not an object")
    typ = str(edit.get("edit_type") or meta.get("augmentation_type") or "")
    if "shift_instead_of_hold" not in typ:
        raise ValueError(f"Expected shift_instead_of_hold edit, found {typ!r}")
    return edit


def _delta_label(delta_ms: int) -> str:
    sign = "m" if int(delta_ms) < 0 else "p"
    return f"delta{sign}{abs(int(delta_ms)):04d}ms"


def generate_shift_hold_insert_time_sweep(
    *,
    base_manifest: Path | None,
    base_json: Path | None,
    base_wav: Path | None,
    out_root: Path,
    split: str,
    window_ms: int,
    step_ms: int,
    min_delta_ms: int | None = None,
    max_delta_ms: int | None = None,
) -> None:
    """Create candidates by moving an existing shift_instead_of_hold insertion.

    The inserted audio block is copied from the base edited WAV and the paired
    natural reference is reused. The only intended variable is insertion time.
    """
    if step_ms <= 0:
        raise ValueError("--step-ms must be positive")
    if window_ms < 0:
        raise ValueError("--window-ms must be non-negative")

    if base_manifest is not None:
        base_rows = read_manifest(base_manifest)
    else:
        if base_json is None or base_wav is None:
            raise ValueError("Provide either --base-manifest or both --base-json and --base-wav")
        base_rows = [{
            "id": base_json.stem,
            "session_id": base_json.stem,
            "audio_path": str(base_wav),
            "json_path": str(base_json),
            "language": "en",
            "session_type": "dyadic",
            "split": split,
        }]

    rows: list[dict[str, Any]] = []
    for row in base_rows:
        json_path = Path(row["json_path"])
        wav_path = Path(row["audio_path"])
        meta = _read_json(json_path)
        edit = _first_shift_hold_edit(meta)
        natural_wav = meta.get("natural_reference_wav")
        natural_json = meta.get("natural_reference_json", "")
        if not natural_wav:
            raise ValueError(f"{json_path} has no natural_reference_wav")

        insert_ms = float(edit["insert_ms"])
        insert_duration_ms = float(edit.get("insert_duration_ms") or (float(edit["insert_end_ms"]) - insert_ms))
        if insert_duration_ms <= 0:
            raise ValueError(f"Invalid insert duration in {json_path}: {insert_duration_ms}")

        natural_audio, sr_nat = sf.read(str(natural_wav), always_2d=True)
        edited_audio, sr_edit = sf.read(str(wav_path), always_2d=True)
        if sr_nat != sr_edit:
            raise ValueError(f"Sample-rate mismatch: natural={sr_nat}, edited={sr_edit}")
        natural_audio = natural_audio.astype(np.float32)
        edited_audio = edited_audio.astype(np.float32)
        if natural_audio.shape[1] == 1:
            natural_audio = np.repeat(natural_audio, 2, axis=1)
        if edited_audio.shape[1] == 1:
            edited_audio = np.repeat(edited_audio, 2, axis=1)
        natural_audio = natural_audio[:, :2]
        edited_audio = edited_audio[:, :2]

        insert_start = int(round(insert_ms * sr_nat / 1000.0))
        insert_len = int(round(insert_duration_ms * sr_nat / 1000.0))
        insert_block = edited_audio[insert_start:insert_start + insert_len].copy()
        if len(insert_block) != insert_len:
            raise ValueError(f"Could not extract full inserted block from {wav_path}")

        base_session = str(meta.get("session_id") or row.get("session_id") or wav_path.stem)
        base_source_stem = str(meta.get("source_natural_stem") or base_session)
        lo_delta = -int(window_ms) if min_delta_ms is None else int(min_delta_ms)
        hi_delta = int(window_ms) if max_delta_ms is None else int(max_delta_ms)
        if hi_delta < lo_delta:
            raise ValueError(f"Invalid sweep range: min_delta_ms={lo_delta} > max_delta_ms={hi_delta}")
        deltas = list(range(lo_delta, hi_delta + 1, int(step_ms)))
        for delta_ms in deltas:
            new_insert_ms = insert_ms + float(delta_ms)
            new_insert = int(round(new_insert_ms * sr_nat / 1000.0))
            if new_insert < 0 or new_insert > len(natural_audio):
                continue
            label = _delta_label(delta_ms)
            stem = f"{base_session}__shift_insert_{label}"
            candidate = np.concatenate(
                [natural_audio[:new_insert], insert_block, natural_audio[new_insert:]],
                axis=0,
            ).astype(np.float32)

            wav_dir = out_root / split / "wav"
            json_dir = out_root / split / "json"
            wav_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)
            out_wav = wav_dir / f"{stem}.wav"
            out_json = json_dir / f"{stem}.json"
            sf.write(str(out_wav), candidate, sr_nat, subtype="FLOAT")

            cand_meta = json.loads(json.dumps(meta))
            cand_meta["session_id"] = stem
            cand_meta["duration"] = float(len(candidate) / sr_nat)
            cand_meta["augmentation_type"] = "shift_instead_of_hold_insert_time_sweep"
            cand_meta["source_natural_stem"] = f"{base_source_stem}__shift_insert_{label}"
            cand_meta["natural_reference_wav"] = str(Path(natural_wav).resolve())
            cand_meta["natural_reference_json"] = str(Path(natural_json).resolve()) if natural_json else ""
            cand_meta["natural_reference_mode"] = "fixed_base_natural_for_insert_time_sweep"
            cand_meta["insert_time_sweep"] = {
                "base_json": str(json_path.resolve()),
                "base_wav": str(wav_path.resolve()),
                "base_natural_wav": str(Path(natural_wav).resolve()),
                "base_insert_ms": float(insert_ms),
                "new_insert_ms": float(new_insert_ms),
                "insert_delta_ms": float(delta_ms),
                "insert_duration_ms": float(insert_duration_ms),
                "step_ms": int(step_ms),
                "window_ms": int(window_ms),
                "min_delta_ms": int(lo_delta),
                "max_delta_ms": int(hi_delta),
                "generation_note": "insert-time sweep candidate generated without model scores",
            }
            cand_edit_meta = cand_meta.get("edit_meta", {})
            if isinstance(cand_edit_meta, dict):
                cand_edit_meta["edit_type"] = "shift_instead_of_hold_insert_time_sweep"
                cand_edits = cand_edit_meta.get("edits", [])
                if isinstance(cand_edits, list) and cand_edits and isinstance(cand_edits[0], dict):
                    cand_edit = cand_edits[0]
                    cand_edit["edit_type"] = "shift_instead_of_hold_insert_time_sweep"
                    cand_edit["insert_ms"] = float(new_insert_ms)
                    cand_edit["insert_end_ms"] = float(new_insert_ms + insert_duration_ms)
                    cand_edit["insert_delta_ms"] = float(delta_ms)
                    cand_edit["insert_position"] = "insert_time_sweep_from_base_shift_instead_of_hold"
                    cand_edit["insert_block_reused_from_base_edit"] = True
                    cand_edit["base_insert_ms"] = float(insert_ms)
                    cand_edit["base_insert_end_ms"] = float(insert_ms + insert_duration_ms)
            write_json(out_json, cand_meta)
            rows.append(manifest_row(stem, out_wav, out_json, split))

    manifest_dir = out_root / "manifests"
    write_manifest(manifest_dir / f"{split}.csv", rows)
    write_manifest(manifest_dir / f"{split}_all_sessions.csv", rows)
    print(f"Wrote {len(rows)} shift-hold insert-time candidates -> {manifest_dir / f'{split}.csv'}")


def run_splice_self_tests() -> None:
    sr = 1000
    n = 100
    a = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    b = np.linspace(0.5, -0.5, n, dtype=np.float32)

    for start in (0, 1, 10, n - 1, n):
        x, y = shift_pair_later(
            a,
            b,
            start,
            7,
            fill=(np.array([0.1, -0.1], dtype=np.float32), np.array([0.0], dtype=np.float32)),
            sr=sr,
        )
        assert len(x) == n + 7 and len(y) == n + 7
        assert x.dtype == np.float32 and y.dtype == np.float32

    x, y = shift_pair_later(a, b, 50, 13, fill=None, sr=sr)
    assert len(x) == n + 13 and len(y) == n + 13
    assert x.dtype == np.float32 and y.dtype == np.float32

    for cut_start, cut_len in ((0, 7), (1, 7), (10, 7), (n - 8, 7)):
        x, y = shift_pair_earlier(a, b, cut_start, cut_len, sr=sr)
        assert len(x) == n - cut_len and len(y) == n - cut_len
        assert x.dtype == np.float32 and y.dtype == np.float32

    x, y = shift_pair_earlier(a.astype(np.float64), b.astype(np.float64), n, 5, sr=sr)
    assert len(x) == n and len(y) == n
    assert x.dtype == np.float32 and y.dtype == np.float32
    print("splice self-tests passed")

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sample-natural")
    s1.add_argument("--csv", required=True)
    s1.add_argument("--out-root", required=True)
    s1.add_argument("--n-pairs", type=int, default=60)
    s1.add_argument("--split", default="natural")
    s1.add_argument("--seed", type=int, default=42)

    s2 = sub.add_parser("make-unnatural")
    s2.add_argument("--natural-csv", required=True)
    s2.add_argument("--out-root", required=True)
    s2.add_argument("--split", default="unnatural_blockwise")
    s2.add_argument("--per-type", type=int, default=10)
    s2.add_argument("--num-shards", type=int, default=1,
                    help="Split source rows deterministically across independent workers. Use a separate --out-root for every shard.")
    s2.add_argument("--shard-index", type=int, default=0,
                    help="Zero-based worker index in [0, num_shards). Rows are assigned by original row index modulo num_shards.")
    s2.add_argument("--seed", type=int, default=43)
    s2.add_argument("--frame-ms", type=float, default=80.0)
    s2.add_argument("--threshold", type=float, default=0.015)
    s2.add_argument("--min-delay-ms", type=int, default=1200)
    s2.add_argument("--max-delay-ms", type=int, default=2000)
    s2.add_argument("--min-advance-ms", type=int, default=1200)
    s2.add_argument("--max-advance-ms", type=int, default=2500)
    s2.add_argument("--max-tries-per-type", type=int, default=200)
    s2.add_argument("--edits-per-sample", type=int, default=1)
    s2.add_argument("--edit-counts", default=None, help="Comma-separated counts, e.g. 1,3,5; keeps same natural sources aligned across counts")
    s2.add_argument("--max-tries-per-sample", type=int, default=40)
    s2.add_argument("--hold-extension-ms", type=int, default=800)
    s2.add_argument("--hold-shift-remove-pad-ms", type=float, default=100.0,
                    help="Room-tone replacement pad around the removed responder turn for hold_instead_of_shift, clamped to neighboring turn boundaries.")
    s2.add_argument("--hold-shift-max-gap-ms", type=float, default=-1.0,
                    help="Optional maximum mutual-silence gap for the original A->B shift in hold_instead_of_shift. Default -1 disables this per-transition cap; use edited hold-gap constraints instead.")
    s2.add_argument("--hold-shift-min-edited-hold-gap-ms", type=float, default=500.0,
                    help="Minimum A-offset to next-A-onset gap after deleting the responder in hold_instead_of_shift.")
    s2.add_argument("--hold-shift-max-edited-hold-gap-ms", type=float, default=1500.0,
                    help="Maximum A-offset to next-A-onset gap after deleting the responder in hold_instead_of_shift; use -1 to disable.")
    s2.add_argument("--hold-shift-min-responder-s", type=float, default=1.2,
                    help="Minimum duration of the removed responder turn for hold_instead_of_shift.")
    s2.add_argument("--hold-shift-max-responder-s", type=float, default=8.0,
                    help="Maximum duration of the removed responder turn for hold_instead_of_shift.")
    s2.add_argument("--hold-shift-require-return-proper", action="store_true",
                    help="Also require the original B->A return to be a strict proper transition. Default off; edited hold-gap constraints are usually cleaner.")
    s2.add_argument("--shift-insert-gain", type=float, default=0.9)
    s2.add_argument("--shift-hold-pre-silence-ms", type=float, default=1000.0,
                    help="Legacy/no-op for shift_instead_of_hold. The 1s pre-offset active-alone guard is enforced by the proper HOLD detector; insertion uses the midpoint of the mutual-silence gap.")
    s2.add_argument("--shift-hold-post-silence-ms", type=float, default=1000.0,
                    help="Legacy/no-op for shift_instead_of_hold. The 1s post-onset active-alone guard is enforced by the proper HOLD detector; insertion uses the midpoint of the mutual-silence gap.")
    s2.add_argument("--shift-hold-min-gap-ms", type=float, default=300.0,
                    help="Minimum original A-A hold gap used by shift_instead_of_hold.")
    s2.add_argument("--shift-hold-max-gap-ms", type=float, default=1000.0,
                    help="Maximum original A-A hold gap used by shift_instead_of_hold before inserting the shift turn.")
    s2.add_argument("--shift-hold-min-insert-s", type=float, default=1.0,
                    help="Minimum duration of the inserted shift turn for shift_instead_of_hold.")
    s2.add_argument("--shift-hold-max-insert-s", type=float, default=4.0,
                    help="Maximum duration of the inserted shift turn for shift_instead_of_hold.")
    s2.add_argument("--bc-insert-gain", type=float, default=1.0)
    s2.add_argument("--bc-insert-count", type=int, default=2,
                    help="Number of copied backchannels inserted by one excessive_backchannel edit.")
    s2.add_argument("--bc-insert-min-gap-ms", type=float, default=800.0,
                    help="Minimum spacing between inserted excessive backchannels.")
    s2.add_argument("--bc-remove-count", type=int, default=1,
                    help="Number of isolated backchannels removed by one missed_backchannel edit.")
    s2.add_argument("--min-missed-backchannels", type=int, default=1,
                    help="Require at least this many isolated backchannels before generating missed_backchannel.")
    s2.add_argument("--allow-bc-source-reuse", action=argparse.BooleanOptionalAction, default=False,
                    help="Allow excessive_backchannel to reuse source BC clips within --max-bc-source-reuse-per-clip. Default off keeps inserted BCs distinct when enough sources exist.")
    s2.add_argument("--max-bc-source-reuse-per-clip", type=int, default=0,
                    help="Maximum number of times each source BC clip may be reused beyond its first use.")
    s2.add_argument("--allow-empty-bc-text", action=argparse.BooleanOptionalAction, default=False,
                    help="Allow VAD-only backchannel clips with empty transcript text. Default off for auditable excessive_backchannel metadata.")
    s2.add_argument("--turn-source", choices=["metadata", "metadata_vad", "silero", "rms"], default="silero",
                    help="Candidate turn source for unnatural edits. silero uses neural VAD boundaries plus transcript text; metadata uses sibling transcript/vad JSON; rms restores old RMS-VAD behavior.")
    s2.add_argument("--silero-threshold", type=float, default=SILERO_CONFIG["threshold"])
    s2.add_argument("--silero-min-speech-ms", type=float, default=SILERO_CONFIG["min_speech_ms"])
    s2.add_argument("--silero-min-silence-ms", type=float, default=SILERO_CONFIG["min_silence_ms"])
    s2.add_argument("--silero-speech-pad-ms", type=float, default=SILERO_CONFIG["speech_pad_ms"])
    s2.add_argument("--silero-merge-gap-ms", type=float, default=SILERO_CONFIG["merge_gap_ms"],
                    help="Merge same-channel Silero speech chunks separated by this many ms.")
    s2.add_argument("--short-context", action="store_true",
                    help="Crop each generated sample around the edited boundary instead of saving the full conversation.")
    s2.add_argument("--min-context-s", type=float, default=20.0)
    s2.add_argument("--max-context-s", type=float, default=25.0)
    s2.add_argument("--late-response-local-context-s", type=float, default=None,
                    help="If set, late_response short-context crops use this local target length while still preserving the delayed response. Keeps strong late clips local instead of forcing 20-25s contexts.")
    s2.add_argument("--late-response-pre-s", type=float, default=4.0,
                    help="Extra pre-anchor budget used when computing the minimum safe local late_response crop length.")
    s2.add_argument("--late-response-post-s", type=float, default=1.0,
                    help="Extra post-response budget used when computing the minimum safe local late_response crop length.")
    s2.add_argument("--scorer-context-s", type=float, default=3.0,
                    help="Minimum past context required by the scorer before the earliest protected turn-taking boundary.")
    s2.add_argument("--scorer-future-s", type=float, default=2.0,
                    help="Future prediction horizon required after the latest protected turn-taking boundary.")
    s2.add_argument("--scorer-unit-pre-s", type=float, default=2.0,
                    help="Pre-boundary unit window used by the scorer; crop guard keeps this whole window after scorer-context-s.")
    s2.add_argument("--scorer-unit-post-s", type=float, default=0.0,
                    help="Post-boundary unit window used by the scorer.")
    s2.add_argument("--enforce-scorer-coverage", action=argparse.BooleanOptionalAction, default=True,
                    help="Require short-context crops to include scorer context before protected boundaries and future horizon after them.")
    s2.add_argument("--require-complete-protected-units", action=argparse.BooleanOptionalAction, default=False,
                    help="Legacy guard: reject edit candidates whose metadata-defined protected boundaries cannot form complete scorer units. Default off because the active scorer no longer uses protected units.")
    s2.add_argument("--types", default=",".join(UNNATURAL_TYPES),
                    help="Comma-separated edit types to generate; add 'interruption' to split strong early overlap from mild early_entry.")
    s2.add_argument("--no-normalize-audio", dest="normalize_audio", action="store_false",
                    help="Disable default RMS/peak normalization for generated WAVs.")
    s2.add_argument("--target-rms-dbfs", type=float, default=-20.0)
    s2.add_argument("--peak-dbfs", type=float, default=-1.0)
    s2.add_argument("--use-input-audio-as-source", action="store_true",
                    help="When natural_csv is a manifest, edit the row audio_path directly instead of rebuilding from participant raw files. Use this for canonical crop-first experiments.")
    s2.add_argument("--reuse-input-audio-as-natural-reference", action="store_true",
                    help="Set natural_reference_wav/json to the input row audio_path/json_path instead of saving a new crop. Requires --use-input-audio-as-source for paired canonical baselines.")
    s2.set_defaults(normalize_audio=True)

    s3 = sub.add_parser("export-vad")
    s3.add_argument("--wav", required=True)
    s3.add_argument("--output", required=True)
    s3.add_argument("--frame-ms", type=float, default=80.0)
    s3.add_argument("--silero-threshold", type=float, default=SILERO_CONFIG["threshold"])
    s3.add_argument("--silero-min-speech-ms", type=float, default=SILERO_CONFIG["min_speech_ms"])
    s3.add_argument("--silero-min-silence-ms", type=float, default=SILERO_CONFIG["min_silence_ms"])
    s3.add_argument("--silero-speech-pad-ms", type=float, default=SILERO_CONFIG["speech_pad_ms"])
    s3.add_argument("--silero-merge-gap-ms", type=float, default=SILERO_CONFIG["merge_gap_ms"])

    s4 = sub.add_parser("shift-hold-insert-sweep")
    s4.add_argument("--base-manifest", default=None,
                    help="Manifest of existing shift_instead_of_hold samples to expand into insert-time candidates.")
    s4.add_argument("--base-json", default=None,
                    help="Single existing shift_instead_of_hold JSON. Use with --base-wav if --base-manifest is not set.")
    s4.add_argument("--base-wav", default=None,
                    help="Single existing shift_instead_of_hold WAV. Use with --base-json if --base-manifest is not set.")
    s4.add_argument("--out-root", required=True)
    s4.add_argument("--split", default="shift_hold_insert_sweep")
    s4.add_argument("--window-ms", type=int, default=2000,
                    help="Symmetric fallback sweep window if --min-delta-ms/--max-delta-ms are not set.")
    s4.add_argument("--min-delta-ms", type=int, default=None,
                    help="Minimum insertion-time offset from the original insert_ms, e.g. -500.")
    s4.add_argument("--max-delta-ms", type=int, default=None,
                    help="Maximum insertion-time offset from the original insert_ms, e.g. 300.")
    s4.add_argument("--step-ms", type=int, default=100,
                    help="Insertion-time sweep step in ms.")


    sub.add_parser("self-test-splice")

    args = ap.parse_args()
    if hasattr(args, "silero_threshold"):
        SILERO_CONFIG.update({
            "threshold": float(args.silero_threshold),
            "min_speech_ms": float(args.silero_min_speech_ms),
            "min_silence_ms": float(args.silero_min_silence_ms),
            "speech_pad_ms": float(args.silero_speech_pad_ms),
            "merge_gap_ms": float(args.silero_merge_gap_ms),
        })
    if args.cmd == "sample-natural":
        sample_natural(
            csv_path=Path(args.csv),
            out_root=Path(args.out_root),
            n_pairs=args.n_pairs,
            split=args.split,
            seed=args.seed,
        )
    elif args.cmd == "export-vad":
        export_vad_labels(
            wav_path=Path(args.wav),
            output_path=Path(args.output),
            frame_ms=args.frame_ms,
        )
    elif args.cmd == "self-test-splice":
        run_splice_self_tests()
    elif args.cmd == "shift-hold-insert-sweep":
        generate_shift_hold_insert_time_sweep(
            base_manifest=Path(args.base_manifest) if args.base_manifest else None,
            base_json=Path(args.base_json) if args.base_json else None,
            base_wav=Path(args.base_wav) if args.base_wav else None,
            out_root=Path(args.out_root),
            split=args.split,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            min_delta_ms=args.min_delta_ms,
            max_delta_ms=args.max_delta_ms,
        )
    else:
        make_unnatural(
            natural_csv=Path(args.natural_csv),
            out_root=Path(args.out_root),
            split=args.split,
            per_type=args.per_type,
            seed=args.seed,
            frame_ms=args.frame_ms,
            threshold=args.threshold,
            min_delay_ms=args.min_delay_ms,
            max_delay_ms=args.max_delay_ms,
            min_advance_ms=args.min_advance_ms,
            max_advance_ms=args.max_advance_ms,
            max_tries_per_type=args.max_tries_per_type,
            edits_per_sample=args.edits_per_sample,
            max_tries_per_sample=args.max_tries_per_sample,
            edit_counts=args.edit_counts,
            hold_extension_ms=args.hold_extension_ms,
            hold_shift_remove_pad_ms=args.hold_shift_remove_pad_ms,
            hold_shift_max_gap_ms=(None if args.hold_shift_max_gap_ms is not None and args.hold_shift_max_gap_ms < 0 else args.hold_shift_max_gap_ms),
            hold_shift_min_edited_hold_gap_ms=args.hold_shift_min_edited_hold_gap_ms,
            hold_shift_max_edited_hold_gap_ms=(None if args.hold_shift_max_edited_hold_gap_ms is not None and args.hold_shift_max_edited_hold_gap_ms < 0 else args.hold_shift_max_edited_hold_gap_ms),
            hold_shift_min_responder_s=args.hold_shift_min_responder_s,
            hold_shift_max_responder_s=args.hold_shift_max_responder_s,
            hold_shift_require_return_proper=args.hold_shift_require_return_proper,
            shift_insert_gain=args.shift_insert_gain,
            shift_hold_pre_silence_ms=args.shift_hold_pre_silence_ms,
            shift_hold_post_silence_ms=args.shift_hold_post_silence_ms,
            shift_hold_min_gap_ms=args.shift_hold_min_gap_ms,
            shift_hold_max_gap_ms=args.shift_hold_max_gap_ms,
            shift_hold_min_insert_s=args.shift_hold_min_insert_s,
            shift_hold_max_insert_s=args.shift_hold_max_insert_s,
            bc_insert_gain=args.bc_insert_gain,
            bc_insert_count=args.bc_insert_count,
            bc_insert_min_gap_ms=args.bc_insert_min_gap_ms,
            bc_remove_count=args.bc_remove_count,
            min_missed_backchannels=args.min_missed_backchannels,
            allow_bc_source_reuse=args.allow_bc_source_reuse,
            max_bc_source_reuse_per_clip=args.max_bc_source_reuse_per_clip,
            allow_empty_bc_text=args.allow_empty_bc_text,
            turn_source=args.turn_source,
            short_context=args.short_context,
            min_context_s=args.min_context_s,
            max_context_s=args.max_context_s,
            late_response_local_context_s=args.late_response_local_context_s,
            late_response_pre_s=args.late_response_pre_s,
            late_response_post_s=args.late_response_post_s,
            scorer_context_s=args.scorer_context_s,
            scorer_future_s=args.scorer_future_s,
            scorer_unit_pre_s=args.scorer_unit_pre_s,
            scorer_unit_post_s=args.scorer_unit_post_s,
            enforce_scorer_coverage=args.enforce_scorer_coverage,
            require_complete_protected_units=args.require_complete_protected_units,
            normalize_audio=args.normalize_audio,
            target_rms_dbfs=args.target_rms_dbfs,
            peak_dbfs=args.peak_dbfs,
            types=[x.strip() for x in args.types.split(",") if x.strip()],
            use_input_audio_as_source=args.use_input_audio_as_source,
            reuse_input_audio_as_natural_reference=args.reuse_input_audio_as_natural_reference,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )


if __name__ == "__main__":
    main()
