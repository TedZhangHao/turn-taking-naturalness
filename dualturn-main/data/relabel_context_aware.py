"""
Context-aware relabeling: derive EOT/HOLD/BOT/BC from dual-channel VAD.

Replaces old utterance-level start/end/shift labels with turn-level labels
that distinguish actual turn boundaries from pauses and backchannels.

See docs/LABEL_REDESIGN.md for full rationale.

Usage:
    python -m dualturn.data.relabel_context_aware                           # both datasets
    python -m dualturn.data.relabel_context_aware --dataset otospeech       # otoSpeech only
    python -m dualturn.data.relabel_context_aware --dataset switchboard     # Switchboard only
    python -m dualturn.data.relabel_context_aware --dry-run                 # stats only, no write
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
try:
    from .path_utils import project_data_root

    PROJECT_DATA_ROOT = project_data_root()
except ImportError:
    # The portable repository only uses compute_context_aware_labels.
    PROJECT_DATA_ROOT = Path("data")

FRAME_RATE = 12.5  # Hz (Mimi codec)
FRAME_MS = 1000.0 / FRAME_RATE  # 80ms per frame


def get_speech_segments(vad: np.ndarray) -> list[tuple[int, int]]:
    """Extract contiguous speech segments as (start, end) frame pairs.
    end is exclusive: vad[start:end] are all 1."""
    segments = []
    in_speech = False
    start = 0
    for t in range(len(vad)):
        if vad[t] > 0.5 and not in_speech:
            start = t
            in_speech = True
        elif vad[t] <= 0.5 and in_speech:
            segments.append((start, t))
            in_speech = False
    if in_speech:
        segments.append((start, len(vad)))
    return segments


def compute_context_aware_labels(
    vad_ch0: np.ndarray,
    vad_ch1: np.ndarray,
    min_turn_frames: int = 12,   # 1.0s -- minimum segment to be a "turn" (not BC)
    max_bc_duration: int = 12,   # 1.0s -- max BC segment duration
    min_bc_silence: int = 12,    # 1.0s -- min silence before/after BC
    max_gap_frames: int = 50,    # 4.0s -- max silence gap for EOT look-ahead
) -> dict[str, np.ndarray]:
    """
    Compute EOT, HOLD, BOT, BC labels from dual-channel VAD.

    Returns dict with keys:
        eot_ch0, eot_ch1   -- End of Turn (sparse, at speech offset frames)
        hold_ch0, hold_ch1 -- Turn Hold / Pause (sparse, at speech offset frames)
        bot_ch0, bot_ch1   -- Beginning of Turn (sparse, at speech onset frames)
        bc_ch0, bc_ch1     -- Backchannel (semi-dense, all frames within BC segment)
    """
    T = len(vad_ch0)

    eot = [np.zeros(T, dtype=np.int8), np.zeros(T, dtype=np.int8)]
    hold = [np.zeros(T, dtype=np.int8), np.zeros(T, dtype=np.int8)]
    bot = [np.zeros(T, dtype=np.int8), np.zeros(T, dtype=np.int8)]
    bc = [np.zeros(T, dtype=np.int8), np.zeros(T, dtype=np.int8)]

    vads = [vad_ch0, vad_ch1]

    for ch in range(2):
        other = 1 - ch
        segments_self = get_speech_segments(vads[ch])
        segments_other = get_speech_segments(vads[other])

        if len(segments_self) == 0:
            continue

        # --- BC detection ---
        # Precompute: distance to previous/next speech onset for this channel
        # so we avoid per-frame while-loops
        vad_binary = (vads[ch] > 0.5).astype(np.int8)
        for seg_start, seg_end in segments_self:
            seg_dur = seg_end - seg_start
            if seg_dur > max_bc_duration:
                continue

            # Check silence before: find previous speech frame
            if seg_start > 0:
                prev_speech = np.where(vad_binary[:seg_start] > 0)[0]
                silence_before = seg_start - (prev_speech[-1] + 1) if len(prev_speech) > 0 else seg_start
            else:
                silence_before = 0
            if silence_before < min_bc_silence:
                continue

            # Check silence after: find next speech frame
            if seg_end < T:
                next_speech = np.where(vad_binary[seg_end:] > 0)[0]
                silence_after = next_speech[0] if len(next_speech) > 0 else (T - seg_end)
            else:
                silence_after = 0
            if silence_after < min_bc_silence:
                continue

            # Other speaker must hold the floor (have a real turn) near this segment
            context_start = max(0, seg_start - max_gap_frames)
            context_end = min(T, seg_end + max_gap_frames)
            other_holds_floor = False
            for os_start, os_end in segments_other:
                if os_end - os_start < min_turn_frames:
                    continue
                if os_end > context_start and os_start < context_end:
                    other_holds_floor = True
                    break
            if other_holds_floor:
                bc[ch][seg_start:seg_end] = 1

        # --- EOT / HOLD at speech offsets ---
        other_binary = (vads[other] > 0.5).astype(np.int8)
        self_binary = vad_binary

        for i, (seg_start, seg_end) in enumerate(segments_self):
            if seg_end >= T:
                continue

            offset_frame = seg_end - 1

            if bc[ch][seg_start:seg_end].any():
                continue

            # Look ahead in the gap region for first activity
            gap_end = min(T, seg_end + max_gap_frames)
            gap_region_other = other_binary[seg_end:gap_end]
            gap_region_self = self_binary[seg_end:gap_end]

            # Find first active frame for other and self in gap
            other_onset_idx = np.where(gap_region_other > 0)[0]
            self_onset_idx = np.where(gap_region_self > 0)[0]

            gap_len = gap_end - seg_end
            first_other = other_onset_idx[0] if len(other_onset_idx) > 0 else gap_len
            first_self = self_onset_idx[0] if len(self_onset_idx) > 0 else gap_len

            other_takes_floor = False
            if first_other < first_self and len(other_onset_idx) > 0:
                abs_start = seg_end + first_other
                abs_end = abs_start
                while abs_end < T and other_binary[abs_end] > 0:
                    abs_end += 1
                seg_true_start = abs_start
                while seg_true_start > 0 and other_binary[seg_true_start - 1] > 0:
                    seg_true_start -= 1
                if abs_end - seg_true_start >= min_turn_frames:
                    other_takes_floor = True

            if other_takes_floor:
                eot[ch][offset_frame] = 1
            else:
                hold[ch][offset_frame] = 1

        # --- BOT at speech onsets ---
        for i, (seg_start, seg_end) in enumerate(segments_self):
            if bc[ch][seg_start:seg_end].any():
                continue

            seg_dur = seg_end - seg_start
            if seg_dur < min_turn_frames:
                continue

            # Look back for previous activity
            lookback_start = max(0, seg_start - max_gap_frames)
            lookback_other = other_binary[lookback_start:seg_start]
            lookback_self = self_binary[lookback_start:seg_start]

            # Find last active frame for other and self before this onset
            other_last_idx = np.where(lookback_other > 0)[0]
            self_last_idx = np.where(lookback_self > 0)[0]

            last_other = other_last_idx[-1] if len(other_last_idx) > 0 else -1
            last_self = self_last_idx[-1] if len(self_last_idx) > 0 else -1

            other_was_speaking = False
            if last_other > last_self and len(other_last_idx) > 0:
                # Other spoke more recently -- check segment duration
                abs_pos = lookback_start + last_other
                abs_start = abs_pos
                while abs_start > 0 and other_binary[abs_start - 1] > 0:
                    abs_start -= 1
                if abs_pos + 1 - abs_start >= min_turn_frames:
                    other_was_speaking = True

            if other_was_speaking:
                bot[ch][seg_start] = 1

    return {
        "eot_ch0": eot[0], "eot_ch1": eot[1],
        "hold_ch0": hold[0], "hold_ch1": hold[1],
        "bot_ch0": bot[0], "bot_ch1": bot[1],
        "bc_ch0": bc[0], "bc_ch1": bc[1],
    }


def relabel_session_dir(session_dir: Path, dry_run: bool = False) -> dict:
    """Relabel a single npy session directory. Saves label arrays as npy files in-place."""
    vad_ch0_path = session_dir / "vad_ch0.npy"
    vad_ch1_path = session_dir / "vad_ch1.npy"

    if not vad_ch0_path.exists() or not vad_ch1_path.exists():
        return {"skipped": True, "reason": "no_vad"}

    vad_ch0 = np.load(vad_ch0_path)
    vad_ch1 = np.load(vad_ch1_path)

    labels = compute_context_aware_labels(vad_ch0, vad_ch1)

    stats = {
        "T": len(vad_ch0),
        "eot_ch0": int(labels["eot_ch0"].sum()),
        "eot_ch1": int(labels["eot_ch1"].sum()),
        "hold_ch0": int(labels["hold_ch0"].sum()),
        "hold_ch1": int(labels["hold_ch1"].sum()),
        "bot_ch0": int(labels["bot_ch0"].sum()),
        "bot_ch1": int(labels["bot_ch1"].sum()),
        "bc_ch0": int(labels["bc_ch0"].sum()),
        "bc_ch1": int(labels["bc_ch1"].sum()),
    }

    if not dry_run:
        for key, arr in labels.items():
            np.save(session_dir / f"{key}.npy", arr)

    return stats


def process_dataset(dataset_dir: Path, name: str, dry_run: bool = False):
    """Process all npy session directories in a dataset directory."""
    session_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    print(f"\n{'='*70}")
    print(f"Processing {name}: {len(session_dirs)} sessions in {dataset_dir}")
    print(f"{'='*70}", flush=True)

    if len(session_dirs) == 0:
        print("  No session directories found, skipping.")
        return

    totals = defaultdict(int)
    skipped = 0
    import time
    t0 = time.time()

    for i, session_dir in enumerate(session_dirs):
        stats = relabel_session_dir(session_dir, dry_run=dry_run)

        if stats.get("skipped"):
            skipped += 1
            continue

        for k, v in stats.items():
            if isinstance(v, int):
                totals[k] += v

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(session_dirs) - i - 1) / rate
            print(f"  [{i+1}/{len(session_dirs)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining", flush=True)

    elapsed = time.time() - t0
    n_processed = len(session_dirs) - skipped
    print(f"\nDone: {n_processed} files in {elapsed:.1f}s ({elapsed/max(n_processed,1)*1000:.0f}ms/file), {skipped} skipped")
    print(f"\nLabel Statistics (totals across all files):")
    print(f"  {'Label':<15} {'Count':>8} {'Per-file avg':>12}")
    print(f"  {'-'*37}")

    for label in ["eot", "hold", "bot", "bc"]:
        for ch in ["ch0", "ch1"]:
            key = f"{label}_{ch}"
            count = totals.get(key, 0)
            avg = count / max(n_processed, 1)
            print(f"  {key:<15} {count:>8} {avg:>12.1f}")

    if dry_run:
        print(f"\n  [DRY RUN] No files were modified.")


def main():
    parser = argparse.ArgumentParser(description="Context-aware relabeling for turn-taking")
    parser.add_argument("--dataset", choices=["otospeech", "switchboard", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without modifying files")
    args = parser.parse_args()

    if args.dataset in ("otospeech", "both"):
        oto_dir = PROJECT_DATA_ROOT / "otospeech_processed_npy"
        if oto_dir.exists():
            process_dataset(oto_dir, "otoSpeech", dry_run=args.dry_run)
        else:
            print(f"Warning: {oto_dir} does not exist")

    if args.dataset in ("switchboard", "both"):
        swb_dir = PROJECT_DATA_ROOT / "switchboard_processed_npy"
        if swb_dir.exists():
            process_dataset(swb_dir, "Switchboard", dry_run=args.dry_run)
        else:
            print(f"Warning: {swb_dir} does not exist")


if __name__ == "__main__":
    main()
