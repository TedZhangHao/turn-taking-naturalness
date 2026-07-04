#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turnnat.data.io import load_audio_and_meta


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_vad_np(vad: np.ndarray, *, min_speech_ms: int, min_silence_ms: int) -> np.ndarray:
    out = vad.astype(np.int8, copy=True)
    n = len(out)
    min_speech = int(min_speech_ms / 20.0)
    min_silence = int(min_silence_ms / 20.0)

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


def silero_vad_50hz(
    waveform_16k: torch.Tensor,
    vad_model: Any,
    *,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
) -> np.ndarray:
    from silero_vad import get_speech_timestamps

    wav = waveform_16k.detach().cpu().float().contiguous()
    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=16_000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    frame_samples = 320
    n_frames = int(wav.numel() // frame_samples)
    vad = np.zeros((n_frames,), dtype=np.int8)
    for ts in timestamps:
        s = int(ts["start"] // frame_samples)
        e = min(int(ts["end"] // frame_samples), n_frames)
        if s < e:
            vad[s:e] = 1
    return vad


def process_row(row: dict[str, str], args: argparse.Namespace, vad_models: list[Any]) -> dict[str, Any]:
    session_id = row.get("id") or row.get("source_natural_stem") or row.get("session_id")
    if not session_id:
        raise KeyError("Manifest row must contain id or session_id")
    out_path = args.output_dir / f"{session_id}.npz"
    if args.skip_existing and out_path.exists():
        return {"id": session_id, "status": "skipped", "path": str(out_path)}

    audio_16k, sr, _ = load_audio_and_meta(row, 16_000)
    assert sr == 16_000

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [
            pool.submit(
                silero_vad_50hz,
                audio_16k[ch],
                vad_models[ch],
                threshold=args.threshold,
                min_speech_ms=args.silero_min_speech_ms,
                min_silence_ms=args.silero_min_silence_ms,
            )
            for ch in range(2)
        ]
        raw0 = futs[0].result()
        raw1 = futs[1].result()

    n = min(len(raw0), len(raw1))
    raw = np.stack([raw0[:n], raw1[:n]], axis=0).astype(np.int8)
    clean = np.stack([
        clean_vad_np(raw[0], min_speech_ms=args.clean_min_speech_ms, min_silence_ms=args.clean_min_silence_ms),
        clean_vad_np(raw[1], min_speech_ms=args.clean_min_speech_ms, min_silence_ms=args.clean_min_silence_ms),
    ], axis=0).astype(np.int8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        vad_50hz=clean,
        vad_raw_50hz=raw,
        sample_rate=np.array(16_000, dtype=np.int32),
        frame_hz=np.array(50.0, dtype=np.float32),
        duration_sec=np.array(audio_16k.shape[-1] / 16_000.0, dtype=np.float32),
    )
    return {"id": session_id, "status": "ok", "path": str(out_path), "frames_50hz": int(n)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache official DualTurn-style Silero VAD labels per session.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--silero-min-speech-ms", type=int, default=100)
    ap.add_argument("--silero-min-silence-ms", type=int, default=50)
    ap.add_argument("--clean-min-speech-ms", type=int, default=150)
    ap.add_argument("--clean-min-silence-ms", type=int, default=150)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    from silero_vad import load_silero_vad

    vad_models = [load_silero_vad().eval(), load_silero_vad().eval()]
    results = []
    for row in tqdm(rows, desc="silero vad cache"):
        results.append(process_row(row, args, vad_models))

    summary = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "num_rows": len(rows),
        "num_ok": sum(r["status"] == "ok" for r in results),
        "num_skipped": sum(r["status"] == "skipped" for r in results),
        "threshold": args.threshold,
        "silero_min_speech_ms": args.silero_min_speech_ms,
        "silero_min_silence_ms": args.silero_min_silence_ms,
        "clean_min_speech_ms": args.clean_min_speech_ms,
        "clean_min_silence_ms": args.clean_min_silence_ms,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
