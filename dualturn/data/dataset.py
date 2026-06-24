from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dualturn.data.io import load_audio_and_meta
from dualturn.data.labels import derive_action_targets, NO_ACTION
from dualturn.data.manifest import load_manifest
from dualturn.data.vad import derive_signals, rms_vad


@dataclass
class ChunkIndex:
    row_idx: int
    start_sample: int
    valid_num_samples: int


class OtoSpeechChunkDataset(Dataset):
    def __init__(self, manifest_path: str, cfg: dict[str, Any], stage: str, training: bool):
        self.cfg = cfg
        self.stage = stage
        self.training = training
        self.rows = load_manifest(manifest_path)

        self.target_sr = int(cfg["data"]["target_sample_rate"])
        self.frame_len = int(cfg["data"]["samples_per_frame"])

        chunk_sec = float(
            cfg["data"]["chunk_seconds"] if training else cfg["data"].get("eval_chunk_seconds", cfg["data"]["chunk_seconds"])
        )
        self.chunk_samples = int(round(chunk_sec * self.target_sr))

        if self.chunk_samples % self.frame_len != 0:
            raise ValueError("chunk_samples must be divisible by frame_len.")

        self.chunks: list[ChunkIndex] = []

        for i, row in enumerate(self.rows):
            dur = float(row.get("duration_sec") or 0.0)
            if dur <= 0:
                continue

            total_samples = int(round(dur * self.target_sr))

            if total_samples <= self.chunk_samples:
                self.chunks.append(ChunkIndex(i, 0, total_samples))
                continue

            stride = max(self.frame_len, self.chunk_samples // 2) if training else self.chunk_samples
            s = 0
            while s < total_samples:
                valid = min(self.chunk_samples, total_samples - s)
                self.chunks.append(ChunkIndex(i, s, valid))

                if s + self.chunk_samples >= total_samples:
                    break
                s += stride

        if not self.chunks:
            raise RuntimeError(f"No chunks built from {manifest_path}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chunk = self.chunks[idx]
        row = self.rows[chunk.row_idx]

        audio, sr, meta = load_audio_and_meta(row, self.target_sr)

        s = chunk.start_sample
        e = min(audio.shape[-1], s + max(chunk.valid_num_samples, 1))
        x_valid = audio[:, s:e]
        valid_num_samples = x_valid.shape[-1]

        if valid_num_samples < self.chunk_samples:
            x = F.pad(x_valid, (0, self.chunk_samples - valid_num_samples))
        else:
            x = x_valid[:, : self.chunk_samples]

        T = self.chunk_samples // self.frame_len
        valid_frames = min(T, (valid_num_samples + self.frame_len - 1) // self.frame_len)

        sample_mask = torch.zeros(self.chunk_samples, dtype=torch.long)
        sample_mask[:valid_num_samples] = 1

        frame_valid_mask = torch.zeros(T, dtype=torch.float32)
        frame_valid_mask[:valid_frames] = 1.0

        item: dict[str, Any] = {
            "id": row["id"],
            "session_id": row.get("session_id", row["id"]),
            "row_idx": chunk.row_idx,
            "start_sample": chunk.start_sample,
            "valid_num_samples": valid_num_samples,
            "audio": x,
            "sample_mask": sample_mask,
            "frame_valid_mask": frame_valid_mask,
        }

        if self.stage in {"stage2", "eval"}:
            threshold = float(self.cfg["data"]["vad"]["rms_threshold"])
            vad = torch.stack([rms_vad(x[ch], self.frame_len, threshold) for ch in range(2)], dim=0)
            vad[:, valid_frames:] = 0.0

            signals = derive_signals(vad, frame_valid_mask, frame_hz=float(self.cfg["data"]["frame_hz"]))
            action_targets = derive_action_targets(signals, frame_hz=float(self.cfg["data"]["frame_hz"]))
            action_targets[frame_valid_mask < 0.5] = NO_ACTION

            item["signal_targets"] = signals
            item["action_targets"] = action_targets

        return item


def collate_chunks(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["id"] = [x["id"] for x in batch]
    out["session_id"] = [x["session_id"] for x in batch]
    out["row_idx"] = torch.tensor([x["row_idx"] for x in batch], dtype=torch.long)
    out["start_sample"] = torch.tensor([x["start_sample"] for x in batch], dtype=torch.long)
    out["valid_num_samples"] = torch.tensor([x["valid_num_samples"] for x in batch], dtype=torch.long)
    out["audio"] = torch.stack([x["audio"] for x in batch], dim=0)
    out["sample_mask"] = torch.stack([x["sample_mask"] for x in batch], dim=0)
    out["frame_valid_mask"] = torch.stack([x["frame_valid_mask"] for x in batch], dim=0)

    if "signal_targets" in batch[0]:
        sig_names = batch[0]["signal_targets"].keys()
        signals = {}
        for name in sig_names:
            signals[name] = torch.stack([x["signal_targets"][name] for x in batch], dim=0)
        out["signal_targets"] = signals
        out["action_targets"] = torch.stack([x["action_targets"] for x in batch], dim=0)

    return out