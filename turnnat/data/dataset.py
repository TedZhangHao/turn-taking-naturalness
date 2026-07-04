from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from turnnat.data.io import load_audio_and_meta
from turnnat.data.manifest import load_manifest


@dataclass
class ChunkIndex:
    row_idx: int
    start_sample: int
    valid_num_samples: int


class StereoChunkDataset(Dataset):
    """Load two-channel conversations and expose fixed-length audio chunks.

    The dataset intentionally does not derive VAD or turn-action labels. The
    paper pipeline builds a shared 50 Hz Silero VAD label stream separately,
    then adapts that same VAD to VAP and DualTurn target grids in the trainer.
    """

    def __init__(self, manifest_path: str, cfg: dict[str, Any], *, training: bool):
        self.cfg = cfg
        self.training = training
        self.rows = load_manifest(manifest_path)

        self.target_sr = int(cfg["data"]["target_sample_rate"])
        self.frame_len = int(cfg["data"]["samples_per_frame"])

        chunk_sec = float(
            cfg["data"]["chunk_seconds"]
            if training
            else cfg["data"].get("eval_chunk_seconds", cfg["data"]["chunk_seconds"])
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
            start = 0
            while start < total_samples:
                valid = min(self.chunk_samples, total_samples - start)
                self.chunks.append(ChunkIndex(i, start, valid))
                if start + self.chunk_samples >= total_samples:
                    break
                start += stride

        if not self.chunks:
            raise RuntimeError(f"No chunks built from {manifest_path}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chunk = self.chunks[idx]
        row = self.rows[chunk.row_idx]
        audio, _, _ = load_audio_and_meta(row, self.target_sr)

        start = chunk.start_sample
        end = min(audio.shape[-1], start + max(chunk.valid_num_samples, 1))
        x_valid = audio[:, start:end]
        valid_num_samples = x_valid.shape[-1]

        if valid_num_samples < self.chunk_samples:
            audio_chunk = F.pad(x_valid, (0, self.chunk_samples - valid_num_samples))
        else:
            audio_chunk = x_valid[:, : self.chunk_samples]

        total_frames = self.chunk_samples // self.frame_len
        valid_frames = min(total_frames, (valid_num_samples + self.frame_len - 1) // self.frame_len)

        sample_mask = torch.zeros(self.chunk_samples, dtype=torch.long)
        sample_mask[:valid_num_samples] = 1

        frame_valid_mask = torch.zeros(total_frames, dtype=torch.float32)
        frame_valid_mask[:valid_frames] = 1.0

        return {
            "id": row["id"],
            "session_id": row.get("session_id", row["id"]),
            "row_idx": chunk.row_idx,
            "start_sample": chunk.start_sample,
            "valid_num_samples": valid_num_samples,
            "audio": audio_chunk,
            "sample_mask": sample_mask,
            "frame_valid_mask": frame_valid_mask,
        }


def collate_chunks(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": [item["id"] for item in batch],
        "session_id": [item["session_id"] for item in batch],
        "row_idx": torch.tensor([item["row_idx"] for item in batch], dtype=torch.long),
        "start_sample": torch.tensor([item["start_sample"] for item in batch], dtype=torch.long),
        "valid_num_samples": torch.tensor([item["valid_num_samples"] for item in batch], dtype=torch.long),
        "audio": torch.stack([item["audio"] for item in batch], dim=0),
        "sample_mask": torch.stack([item["sample_mask"] for item in batch], dim=0),
        "frame_valid_mask": torch.stack([item["frame_valid_mask"] for item in batch], dim=0),
    }

