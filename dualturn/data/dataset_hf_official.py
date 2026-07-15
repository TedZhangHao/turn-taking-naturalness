from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
from datasets import load_dataset

from dualturn.data.labels import derive_action_targets, NO_ACTION


@dataclass
class HFChunkIndex:
    row_idx: int
    start_frame: int
    valid_num_frames: int


def normalize_official_repo_ids(repo_ids: str | list[str] | None) -> list[str]:
    if repo_ids is None:
        return []
    if isinstance(repo_ids, str):
        return [repo_ids]
    return [str(x) for x in repo_ids]


class HFDualTurnFeatureDataset(Dataset):
    """
    Read the official processed HF dataset directly:
      - mimi_feat_ch0 / mimi_feat_ch1
      - vad/eot/hold/bot/bc/fvad labels

    Important:
      The official HF dataset stores several 2D tensors as FLAT lists:
        - mimi_feat_ch*: [num_frames * 512]
        - fvad_ch*:      [num_frames * 4]

      So we must reshape using row["num_frames"] before chunk slicing.
    """

    def __init__(
        self,
        repo_id: str,
        split: str,
        cfg: dict[str, Any],
        training: bool,
        cache_dir: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.cfg = cfg
        self.training = training
        self.frame_hz = float(cfg["data"]["frame_hz"])
        self.channel_swap_prob = float(cfg["data"].get("channel_swap_prob", 0.0)) if training else 0.0
        self.soft_labels = bool(cfg["data"].get("soft_labels", False))
        self.sigma_before = float(cfg["data"].get("sigma_before", 3.0))
        self.sigma_after = float(cfg["data"].get("sigma_after", 1.0))
        self.event_label_width = int(cfg["data"].get("event_label_width", 1))
        self.fvad_bins = list(cfg.get("stage2", {}).get("fvad_bins", [3, 6, 12, 25]))

        if "window_frames" in cfg["data"]:
            self.chunk_frames = int(cfg["data"]["window_frames"])
        else:
            chunk_sec = float(
                cfg["data"]["chunk_seconds"]
                if training
                else cfg["data"].get("eval_chunk_seconds", cfg["data"]["chunk_seconds"])
            )
            self.chunk_frames = int(round(chunk_sec * self.frame_hz))
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")

        if training:
            self.stride_frames = int(cfg["data"].get("hop_frames_train", max(1, self.chunk_frames // 2)))
        else:
            self.stride_frames = int(cfg["data"].get("hop_frames_val", self.chunk_frames))
        self.stride_frames = max(1, self.stride_frames)

        self.ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)

        self.chunks: list[HFChunkIndex] = []
        for i in range(len(self.ds)):
            row = self.ds[i]
            total_frames = int(row["num_frames"])
            if total_frames <= self.chunk_frames:
                self.chunks.append(HFChunkIndex(i, 0, total_frames))
                continue

            s = 0
            while s < total_frames:
                valid = min(self.chunk_frames, total_frames - s)
                self.chunks.append(HFChunkIndex(i, s, valid))
                if s + self.chunk_frames >= total_frames:
                    break
                s += self.stride_frames

        if not self.chunks:
            raise RuntimeError(f"No chunks built from repo={repo_id} split={split}")

    def __len__(self) -> int:
        return len(self.chunks)

    def _slice_1d(self, arr, s: int, e: int, T: int, dtype=torch.float32):
        x = torch.tensor(arr[s:e], dtype=dtype)
        if x.numel() < T:
            x = F.pad(x, (0, T - x.numel()))
        return x

    def _reshape_flat_2d(self, flat_arr, total_frames: int, D: int, dtype=torch.float32):
        x = torch.tensor(flat_arr, dtype=dtype)
        expected = total_frames * D
        if x.numel() != expected:
            raise RuntimeError(
                f"Expected flat array with {expected} values (= {total_frames}*{D}), "
                f"but got {x.numel()}."
            )
        return x.view(total_frames, D)

    def _slice_flat_2d(self, flat_arr, total_frames: int, s: int, e: int, T: int, D: int, dtype=torch.float32):
        full = self._reshape_flat_2d(flat_arr, total_frames=total_frames, D=D, dtype=dtype)
        x = full[s:e]
        if x.shape[0] < T:
            pad = torch.zeros((T - x.shape[0], D), dtype=dtype)
            x = torch.cat([x, pad], dim=0)
        return x

    def _dilate_binary(self, x: torch.Tensor, width: int) -> torch.Tensor:
        if width <= 1:
            return x
        pad = width // 2
        y = F.max_pool1d(x.unsqueeze(0).unsqueeze(0), kernel_size=width, stride=1, padding=pad)
        return y.squeeze(0).squeeze(0)

    def _smooth_events(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        idxs = torch.nonzero(x > 0.5).reshape(-1)
        T = x.numel()
        for idx in idxs.tolist():
            left = max(0, idx - 12)
            right = min(T, idx + 8)
            for t in range(left, right):
                sigma = self.sigma_before if t <= idx else self.sigma_after
                if sigma <= 0:
                    value = 1.0 if t == idx else 0.0
                else:
                    value = math.exp(-0.5 * ((t - idx) / sigma) ** 2)
                out[t] = torch.maximum(out[t], out.new_tensor(float(value)))
        return out

    def _postprocess_sparse_signal(self, x: torch.Tensor) -> torch.Tensor:
        if self.soft_labels:
            return self._smooth_events(x)
        return self._dilate_binary(x, self.event_label_width)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chunk = self.chunks[idx]
        row = self.ds[chunk.row_idx]
        s = chunk.start_frame
        e = s + chunk.valid_num_frames
        T = self.chunk_frames
        total_frames = int(row["num_frames"])

        feat0 = self._slice_flat_2d(row["mimi_feat_ch0"], total_frames, s, e, T, 512, dtype=torch.float32)
        feat1 = self._slice_flat_2d(row["mimi_feat_ch1"], total_frames, s, e, T, 512, dtype=torch.float32)
        codes0 = self._slice_flat_2d(row["codes_ch0"], total_frames, s, e, T, 8, dtype=torch.long)
        codes1 = self._slice_flat_2d(row["codes_ch1"], total_frames, s, e, T, 8, dtype=torch.long)

        frame_valid_mask = torch.zeros(T, dtype=torch.float32)
        frame_valid_mask[: chunk.valid_num_frames] = 1.0

        signals = {
            "vad": torch.stack([
                self._slice_1d(row["vad_ch0"], s, e, T, dtype=torch.float32),
                self._slice_1d(row["vad_ch1"], s, e, T, dtype=torch.float32),
            ], dim=0),
            "eot": torch.stack([
                self._slice_1d(row["eot_ch0"], s, e, T, dtype=torch.float32),
                self._slice_1d(row["eot_ch1"], s, e, T, dtype=torch.float32),
            ], dim=0),
            "hold": torch.stack([
                self._slice_1d(row["hold_ch0"], s, e, T, dtype=torch.float32),
                self._slice_1d(row["hold_ch1"], s, e, T, dtype=torch.float32),
            ], dim=0),
            "bot": torch.stack([
                self._slice_1d(row["bot_ch0"], s, e, T, dtype=torch.float32),
                self._slice_1d(row["bot_ch1"], s, e, T, dtype=torch.float32),
            ], dim=0),
            "bc": torch.stack([
                self._slice_1d(row["bc_ch0"], s, e, T, dtype=torch.float32),
                self._slice_1d(row["bc_ch1"], s, e, T, dtype=torch.float32),
            ], dim=0),
            "fvad": torch.stack([
                self._slice_flat_2d(row["fvad_ch0"], total_frames, s, e, T, 4, dtype=torch.float32),
                self._slice_flat_2d(row["fvad_ch1"], total_frames, s, e, T, 4, dtype=torch.float32),
            ], dim=0),
            "frame_valid_mask": frame_valid_mask.clone(),
        }

        for name in ["eot", "hold", "bot", "bc"]:
            for ch in range(2):
                signals[name][ch] = self._postprocess_sparse_signal(signals[name][ch])

        max_fvad_bin = max(int(x) for x in self.fvad_bins)
        fvad_mask = torch.zeros(T, dtype=torch.float32)
        valid_fvad = max(0, chunk.valid_num_frames - max_fvad_bin)
        fvad_mask[:valid_fvad] = 1.0

        if self.training and self.channel_swap_prob > 0.0 and torch.rand(()) < self.channel_swap_prob:
            feat0, feat1 = feat1, feat0
            codes0, codes1 = codes1, codes0
            for name in ["vad", "eot", "hold", "bot", "bc", "fvad"]:
                signals[name] = signals[name].flip(0)

        action_targets = derive_action_targets(signals, frame_hz=self.frame_hz)
        action_targets[frame_valid_mask < 0.5] = NO_ACTION

        return {
            "session_id": row["session_id"],
            "dataset": row.get("dataset", ""),
            "codes_ch0": codes0,
            "codes_ch1": codes1,
            "mimi_feat_ch0": feat0,
            "mimi_feat_ch1": feat1,
            "frame_valid_mask": frame_valid_mask,
            "fvad_mask": fvad_mask,
            "signal_targets": signals,
            "action_targets": action_targets,
        }


def collate_hf_feature_chunks(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["session_id"] = [x["session_id"] for x in batch]
    out["dataset"] = [x["dataset"] for x in batch]
    out["codes_ch0"] = torch.stack([x["codes_ch0"] for x in batch], dim=0)
    out["codes_ch1"] = torch.stack([x["codes_ch1"] for x in batch], dim=0)
    out["mimi_feat_ch0"] = torch.stack([x["mimi_feat_ch0"] for x in batch], dim=0)
    out["mimi_feat_ch1"] = torch.stack([x["mimi_feat_ch1"] for x in batch], dim=0)
    out["frame_valid_mask"] = torch.stack([x["frame_valid_mask"] for x in batch], dim=0)
    out["fvad_mask"] = torch.stack([x["fvad_mask"] for x in batch], dim=0)
    out["action_targets"] = torch.stack([x["action_targets"] for x in batch], dim=0)

    sig_names = batch[0]["signal_targets"].keys()
    signals = {}
    for name in sig_names:
        signals[name] = torch.stack([x["signal_targets"][name] for x in batch], dim=0)
    out["signal_targets"] = signals
    return out


def build_official_hf_dataset(
    repo_ids: str | list[str],
    split: str,
    cfg: dict[str, Any],
    training: bool,
    cache_dir: str | None = None,
) -> Dataset:
    repo_list = normalize_official_repo_ids(repo_ids)
    if not repo_list:
        raise ValueError("At least one official HF dataset repo id is required.")

    datasets = [
        HFDualTurnFeatureDataset(
            repo_id=repo_id,
            split=split,
            cfg=cfg,
            training=training,
            cache_dir=cache_dir,
        )
        for repo_id in repo_list
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
