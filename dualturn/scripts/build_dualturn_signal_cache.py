#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DUALTURN_MAIN = PROJECT_ROOT / "dualturn-main"
if str(DUALTURN_MAIN) not in sys.path:
    sys.path.append(str(DUALTURN_MAIN))

from data.relabel_context_aware import compute_context_aware_labels


DEFAULT_BINS = [3, 6, 12, 25]


def read_ids(manifests: list[Path]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sample_id = row.get("id") or row.get("source_natural_stem") or row.get("session_id")
                if not sample_id:
                    raise KeyError(f"Manifest row in {manifest} has no sample ID")
                if sample_id not in seen:
                    seen.add(sample_id)
                    ids.append(str(sample_id))
    return ids


def downsample_vad(vad_50hz: np.ndarray, threshold: float) -> np.ndarray:
    if vad_50hz.ndim != 2 or vad_50hz.shape[0] != 2:
        raise ValueError(f"Expected VAD [2,T50], got {vad_50hz.shape}")
    frames = vad_50hz.shape[1] // 4
    grouped = vad_50hz[:, :frames * 4].reshape(2, frames, 4).mean(axis=-1)
    return (grouped >= threshold).astype(np.uint8)


def compute_fvad(vad: np.ndarray, bin_edges: list[int]) -> tuple[np.ndarray, np.ndarray]:
    channels, frames = vad.shape
    targets = np.zeros((frames, channels * len(bin_edges)), dtype=np.float32)
    mask = np.zeros((frames,), dtype=np.uint8)
    valid_frames = frames - int(bin_edges[-1])
    if valid_frames <= 0:
        return targets, mask

    idx = np.arange(valid_frames)
    for ch in range(channels):
        cumulative = np.cumsum(np.pad(vad[ch].astype(np.float32), (1, 0)), dtype=np.float64)
        previous = 0
        for bin_index, edge in enumerate(bin_edges):
            lo = idx + previous + 1
            hi = idx + int(edge) + 1
            targets[:valid_frames, ch * len(bin_edges) + bin_index] = (
                cumulative[hi] - cumulative[lo]
            ) / float(edge - previous)
            previous = int(edge)
    mask[:valid_frames] = 1
    return targets, mask


def process_one(
    sample_id: str,
    vad_cache_dir: str,
    output_dir: str,
    threshold: float,
    bin_edges: list[int],
    skip_existing: bool,
) -> dict[str, Any]:
    source = Path(vad_cache_dir) / f"{sample_id}.npz"
    destination = Path(output_dir) / f"{sample_id}.npz"
    if skip_existing and destination.is_file():
        return {"id": sample_id, "status": "skipped"}
    if not source.is_file():
        return {"id": sample_id, "status": "missing_vad", "path": str(source)}

    with np.load(source) as data:
        vad_50hz = data["vad_50hz"].astype(np.float32)
    vad = downsample_vad(vad_50hz, threshold)
    context = compute_context_aware_labels(vad[0], vad[1])
    fvad, fvad_mask = compute_fvad(vad, bin_edges)

    payload: dict[str, np.ndarray] = {
        "vad": vad.T.astype(np.uint8),
        "fvad": fvad,
        "fvad_mask": fvad_mask,
        "eot": np.stack([context["eot_ch0"], context["eot_ch1"]], axis=-1).astype(np.uint8),
        "hold": np.stack([context["hold_ch0"], context["hold_ch1"]], axis=-1).astype(np.uint8),
        "bot": np.stack([context["bot_ch0"], context["bot_ch1"]], axis=-1).astype(np.uint8),
        "bc": np.stack([context["bc_ch0"], context["bc_ch1"]], axis=-1).astype(np.uint8),
        "frame_hz": np.array(12.5, dtype=np.float32),
        "source_frame_hz": np.array(50.0, dtype=np.float32),
        "fvad_bin_edges": np.asarray(bin_edges, dtype=np.int16),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return {
        "id": sample_id,
        "status": "ok",
        "frames": int(vad.shape[1]),
        **{f"{name}_positive": int(payload[name].sum()) for name in ["vad", "eot", "hold", "bot", "bc"]},
        "fvad_valid": int(fvad_mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate official DualTurn six-signal labels from cached 50Hz Silero VAD."
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--vad-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vad-downsample-threshold", type=float, default=0.5)
    parser.add_argument("--fvad-bin-edges", default="3,6,12,25")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    bin_edges = [int(x) for x in args.fvad_bin_edges.split(",") if x.strip()]
    if bin_edges != sorted(bin_edges) or not bin_edges or bin_edges[0] <= 0:
        raise ValueError(f"Invalid FVAD bin edges: {bin_edges}")
    sample_ids = read_ids(args.manifest)
    if args.limit is not None:
        sample_ids = sample_ids[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                process_one,
                sample_id,
                str(args.vad_cache_dir),
                str(args.output_dir),
                args.vad_downsample_threshold,
                bin_edges,
                args.skip_existing,
            )
            for sample_id in sample_ids
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="DualTurn signal labels"):
            results.append(future.result())

    statuses = Counter(result["status"] for result in results)
    summary: dict[str, Any] = {
        "manifests": [str(path) for path in args.manifest],
        "vad_cache_dir": str(args.vad_cache_dir),
        "output_dir": str(args.output_dir),
        "num_sessions": len(sample_ids),
        "statuses": dict(statuses),
        "vad_downsample": "mean over each four 50Hz frames, then threshold",
        "vad_downsample_threshold": args.vad_downsample_threshold,
        "frame_hz": 12.5,
        "fvad_bin_edges": bin_edges,
        "label_implementation": str(DUALTURN_MAIN / "data" / "relabel_context_aware.py"),
    }
    for key in ["frames", "vad_positive", "eot_positive", "hold_positive", "bot_positive", "bc_positive", "fvad_valid"]:
        summary[f"total_{key}"] = int(sum(result.get(key, 0) for result in results))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if statuses.get("missing_vad", 0):
        raise SystemExit(f"Missing VAD cache for {statuses['missing_vad']} sessions")


if __name__ == "__main__":
    main()
