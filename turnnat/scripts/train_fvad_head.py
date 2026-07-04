#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAP_ROOT = PROJECT_ROOT / "VAP-main"
for path in [PROJECT_ROOT, VAP_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from turnnat.config import ensure_output_dirs, load_config
from turnnat.data.dataset import StereoChunkDataset, collate_chunks
from turnnat.utils import count_trainable_parameters, save_json, set_seed


DEFAULT_VAP_CKPT = PROJECT_ROOT / "VAP-main" / "example" / "checkpoints" / "VAP_state_dict.pt"
VAP_STATE_DICT_URL = "https://github.com/ErikEkstedt/VAP/raw/main/example/checkpoints/VAP_state_dict.pt"
DEFAULT_DUALTURN_MODEL_ID = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"
DEFAULT_VAP_BIN_TIMES = [0.2, 0.4, 0.6, 0.8]
IGNORE_INDEX = -100


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, out_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAP256Model(nn.Module):
    objective_type = "categorical256"
    multitask = False

    def __init__(
        self,
        ckpt: str | Path | None,
        hidden_dim: int,
        dropout: float,
        head_init: str,
    ) -> None:
        super().__init__()
        from vap.modules.VAP import VAP
        from vap.modules.encoder import EncoderCPC
        from vap.modules.modules import TransformerStereo

        self.backbone = VAP(EncoderCPC(), TransformerStereo())
        if ckpt is not None:
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Missing VAP checkpoint: {ckpt_path}. Download it with:\n"
                    f"  mkdir -p {ckpt_path.parent}\n"
                    f"  curl -L {VAP_STATE_DICT_URL} -o {ckpt_path}"
                )
            try:
                state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(str(ckpt_path), map_location="cpu")
            self.backbone.load_state_dict(state, strict=True)
            print(f"Loaded VAP backbone: {ckpt_path}")

        self.vap_head_256 = ProjectionHead(self.backbone.dim, hidden_dim, dropout, out_dim=256)
        use_pretrained_head = head_init == "pretrained" or (
            head_init == "auto" and ckpt is not None and hidden_dim <= 0
        )
        if use_pretrained_head:
            if ckpt is None:
                raise ValueError("--head-init pretrained requires --vap-ckpt")
            if hidden_dim > 0:
                raise ValueError("The pretrained VAP head is linear; use --head-hidden-dim 0")
            self.vap_head_256.net.load_state_dict(self.backbone.vap_head.state_dict(), strict=True)
            print("Initialized task head from the pretrained VAP vap_head")
        else:
            print("Initialized a new random VAP task head")

    def head_parameters(self):
        return self.vap_head_256.parameters()

    def fvad_head_parameters(self):
        return self.vap_head_256.parameters()

    def forward(self, audio: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.forward_all(audio)["fvad"]

    def forward_all(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(audio)
        return {
            "fvad": self.vap_head_256(out["x"]),
            "vad": out["vad"],
        }


class OfficialDualTurnFVADModel(nn.Module):
    TASKS = ("vad", "fvad", "eot", "bot", "hold", "bc")

    def __init__(
        self,
        model_id: str,
        local_files_only: bool,
        head_init: str,
        *,
        fvad_head_type: str,
        multitask: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list[str],
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if int(getattr(self.model.config, "num_fvad_bins", 4)) != 4:
            raise ValueError("This trainer currently expects the official four-bin DualTurn FVAD head")
        self.multitask = bool(multitask)
        self.task_layer_weights = self._load_task_layer_weights(model_id, local_files_only)
        self.fvad_head_type = str(fvad_head_type)
        if self.fvad_head_type == "categorical256":
            if head_init == "pretrained":
                raise ValueError("DualTurn categorical256 has no pretrained head; use --head-init random")
            self.objective_type = "categorical256"
            self.fvad_head_256 = nn.Linear(int(self.model.fvad_head.in_features), 256)
            print("Initialized a new DualTurn 256-class FVAD head")
        elif self.fvad_head_type == "native8":
            self.objective_type = "bernoulli8"
            self.fvad_head_256 = None
            if head_init == "random":
                self.model.fvad_head.reset_parameters()
                print("Reinitialized the official DualTurn FVAD head")
            else:
                print("Using the pretrained official DualTurn FVAD head")
        else:
            raise ValueError(f"Unsupported DualTurn FVAD head type: {self.fvad_head_type}")
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(lora_target_modules),
                bias="none",
            )
            self.model.backbone = get_peft_model(self.model.backbone, lora_cfg)
            print(
                "Attached fresh LoRA adapters to the checkpoint's merged Qwen backbone: "
                f"r={lora_r}, alpha={lora_alpha}"
            )
        print(f"Loaded official DualTurn backbone: {model_id}")

    def _fvad_head(self) -> nn.Module:
        if self.fvad_head_256 is not None:
            return self.fvad_head_256
        return self.model.fvad_head

    @staticmethod
    def _load_task_layer_weights(model_id: str, local_files_only: bool) -> nn.ParameterDict | None:
        """Restore task-layer attention tensors ignored by the public remote-code wrapper."""
        try:
            from safetensors import safe_open
            from transformers.utils.hub import cached_file

            weights_path = cached_file(
                model_id,
                "model.safetensors",
                local_files_only=local_files_only,
            )
            if weights_path is None:
                return None
            tensors: dict[str, nn.Parameter] = {}
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                for task in OfficialDualTurnFVADModel.TASKS:
                    key = f"task_layer_weights.{task}"
                    if key not in f.keys():
                        return None
                    tensors[task] = nn.Parameter(f.get_tensor(key).float())
            print("Restored pretrained per-task layer-attention weights")
            return nn.ParameterDict(tensors)
        except (ImportError, OSError):
            print("Task-layer weights unavailable; using the final Qwen layer")
            return None

    def head_parameters(self):
        if not self.multitask:
            return self._fvad_head().parameters()
        heads = [
            self.model.vad_head_ch0,
            self.model.vad_head_ch1,
            self._fvad_head(),
            self.model.eot_head_ch0,
            self.model.eot_head_ch1,
            self.model.bot_head_ch0,
            self.model.bot_head_ch1,
            self.model.hold_head_ch0,
            self.model.hold_head_ch1,
            self.model.bc_head_ch0,
            self.model.bc_head_ch1,
        ]
        return (p for head in heads for p in head.parameters())

    def fvad_head_parameters(self):
        return self._fvad_head().parameters()

    def _hidden_one(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        model = self.model
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).repeat(2, 1)
        if audio.dim() != 2 or audio.shape[0] != 2:
            raise ValueError(f"Expected one stereo audio example [2, T], got {tuple(audio.shape)}")

        feat0 = model._encode_channel(audio[0])
        feat1 = model._encode_channel(audio[1])
        T = min(feat0.shape[0], feat1.shape[0])
        mimi_feat_ch0 = feat0[:T].unsqueeze(0)
        mimi_feat_ch1 = feat1[:T].unsqueeze(0)
        embeddings = model.mimi_projection(mimi_feat_ch0, mimi_feat_ch1)
        out = model.backbone(
            inputs_embeds=embeddings,
            output_hidden_states=True,
            return_dict=True,
        )
        final = out.hidden_states[-1]
        if self.task_layer_weights is None:
            return {task: final for task in self.TASKS}

        hidden_stack = torch.stack(tuple(out.hidden_states), dim=0)
        if hidden_stack.shape[0] != next(iter(self.task_layer_weights.values())).numel():
            raise ValueError(
                "DualTurn task-layer weights do not match Qwen hidden-state count: "
                f"{next(iter(self.task_layer_weights.values())).numel()} vs {hidden_stack.shape[0]}"
            )
        return {
            task: (hidden_stack * torch.softmax(self.task_layer_weights[task], dim=0)[:, None, None, None]).sum(dim=0)
            for task in self.TASKS
        }

    def _hidden_batch(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        if audio.dim() == 2:
            return self._hidden_one(audio)
        if audio.dim() != 3 or audio.shape[1] != 2:
            raise ValueError(f"Expected audio [2, T] or [B, 2, T], got {tuple(audio.shape)}")
        per_example = [self._hidden_one(audio[i]) for i in range(audio.shape[0])]
        return {
            task: torch.cat([item[task] for item in per_example], dim=0)
            for task in self.TASKS
        }

    def _hidden_from_features(
        self,
        mimi_feat_ch0: torch.Tensor,
        mimi_feat_ch1: torch.Tensor,
        tasks: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        tasks = tasks or self.TASKS
        embeddings = self.model.mimi_projection(mimi_feat_ch0, mimi_feat_ch1)
        out = self.model.backbone(
            inputs_embeds=embeddings,
            output_hidden_states=True,
            return_dict=True,
        )
        final = out.hidden_states[-1]
        if self.task_layer_weights is None:
            return {task: final for task in tasks}
        hidden_stack = torch.stack(tuple(out.hidden_states), dim=0)
        return {
            task: (hidden_stack * torch.softmax(self.task_layer_weights[task], dim=0)[:, None, None, None]).sum(dim=0)
            for task in tasks
        }

    def forward_fvad(
        self,
        *,
        mimi_feat_ch0: torch.Tensor,
        mimi_feat_ch1: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._hidden_from_features(mimi_feat_ch0, mimi_feat_ch1, tasks=("fvad",))["fvad"]
        return self._fvad_head()(hidden).float()

    def forward_all(
        self,
        audio: torch.Tensor | None = None,
        *,
        mimi_feat_ch0: torch.Tensor | None = None,
        mimi_feat_ch1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mimi_feat_ch0 is not None and mimi_feat_ch1 is not None:
            h = self._hidden_from_features(mimi_feat_ch0, mimi_feat_ch1)
        elif audio is not None:
            h = self._hidden_batch(audio)
        else:
            raise ValueError("Provide audio or both precomputed Mimi feature tensors")
        model = self.model

        def two(head0: nn.Module, head1: nn.Module, task: str) -> torch.Tensor:
            return torch.stack(
                [head0(h[task]).squeeze(-1), head1(h[task]).squeeze(-1)], dim=-1
            ).float()

        return {
            "vad": two(model.vad_head_ch0, model.vad_head_ch1, "vad"),
            "fvad": self._fvad_head()(h["fvad"]).float(),
            "eot": two(model.eot_head_ch0, model.eot_head_ch1, "eot"),
            "bot": two(model.bot_head_ch0, model.bot_head_ch1, "bot"),
            "hold": two(model.hold_head_ch0, model.hold_head_ch1, "hold"),
            "bc": two(model.bc_head_ch0, model.bc_head_ch1, "bc"),
        }

    def forward(
        self,
        audio: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_all(audio)["fvad"]


class MimiFeatureCache:
    def __init__(self, root: Path, *, sample_rate: int = 24_000, frame_hz: float = 12.5) -> None:
        self.root = Path(root)
        self.sample_rate = int(sample_rate)
        self.frame_hz = float(frame_hz)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Missing Mimi feature root: {self.root}")
        self._arrays: dict[tuple[str, int], np.ndarray] = {}

    def batch(self, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        target_frames = int(batch["frame_valid_mask"].shape[1])
        channels: list[list[torch.Tensor]] = [[], []]
        for index, sample_id in enumerate(batch["id"]):
            start_sample = int(batch["start_sample"][index].item())
            start_frame = int(round(start_sample * self.frame_hz / self.sample_rate))
            for ch in range(2):
                path = self.root / str(sample_id) / f"mimi_feat_ch{ch}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing precomputed Mimi feature: {path}")
                key = (str(sample_id), ch)
                if key not in self._arrays:
                    self._arrays[key] = np.load(path, mmap_mode="r")
                values = self._arrays[key]
                sliced = np.asarray(values[start_frame:start_frame + target_frames], dtype=np.float32).copy()
                tensor = torch.from_numpy(sliced)
                if tensor.shape[0] < target_frames:
                    tensor = F.pad(tensor, (0, 0, 0, target_frames - tensor.shape[0]))
                channels[ch].append(tensor)
        return (
            torch.stack(channels[0], dim=0).to(device, non_blocking=True),
            torch.stack(channels[1], dim=0).to(device, non_blocking=True),
        )


class SignalLabelCache:
    SIGNALS = ("vad", "fvad", "eot", "hold", "bot", "bc")

    def __init__(self, root: Path, *, sample_rate: int = 24_000, frame_hz: float = 12.5) -> None:
        self.root = Path(root)
        self.sample_rate = int(sample_rate)
        self.frame_hz = float(frame_hz)
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Missing signal-label cache: {self.root}. Run build_dualturn_signal_cache.py first."
            )

    def _load(self, sample_id: str) -> dict[str, np.ndarray]:
        if sample_id not in self._cache:
            path = self.root / f"{sample_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing signal labels for {sample_id}: {path}")
            with np.load(path) as data:
                missing = [key for key in (*self.SIGNALS, "fvad_mask") if key not in data]
                if missing:
                    raise KeyError(f"Signal cache {path} is missing {missing}")
                self._cache[sample_id] = {
                    key: data[key].copy() for key in (*self.SIGNALS, "fvad_mask")
                }
        return self._cache[sample_id]

    def batch(self, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
        target_frames = int(batch["frame_valid_mask"].shape[1])
        rows: dict[str, list[torch.Tensor]] = {
            key: [] for key in (*self.SIGNALS, "fvad_mask")
        }
        for index, sample_id_value in enumerate(batch["id"]):
            sample_id = str(sample_id_value)
            start_sample = int(batch["start_sample"][index].item())
            start_frame = int(round(start_sample * self.frame_hz / self.sample_rate))
            arrays = self._load(sample_id)
            for key, values in arrays.items():
                sliced = values[start_frame:start_frame + target_frames]
                tensor = torch.from_numpy(np.asarray(sliced, dtype=np.float32).copy())
                if tensor.shape[0] < target_frames:
                    if tensor.dim() == 1:
                        tensor = F.pad(tensor, (0, target_frames - tensor.shape[0]))
                    else:
                        tensor = F.pad(tensor, (0, 0, 0, target_frames - tensor.shape[0]))
                rows[key].append(tensor)
        return {
            key: torch.stack(values, dim=0).to(device, non_blocking=True)
            for key, values in rows.items()
        }


class MimiFeatureChunkDataset(Dataset):
    """Chunk index for precomputed Mimi training without repeatedly reading WAV files."""

    def __init__(self, manifest_path: str, cfg: dict[str, Any], *, training: bool) -> None:
        from turnnat.data.manifest import load_manifest

        self.rows = load_manifest(manifest_path)
        self.sample_rate = int(cfg["data"]["target_sample_rate"])
        self.samples_per_frame = int(cfg["data"]["samples_per_frame"])
        chunk_seconds = float(
            cfg["data"]["chunk_seconds"]
            if training
            else cfg["data"].get("eval_chunk_seconds", cfg["data"]["chunk_seconds"])
        )
        self.chunk_samples = int(round(chunk_seconds * self.sample_rate))
        self.chunks: list[tuple[int, int, int]] = []
        for row_index, row in enumerate(self.rows):
            total_samples = int(round(float(row.get("duration_sec") or 0.0) * self.sample_rate))
            if total_samples <= 0:
                continue
            if total_samples <= self.chunk_samples:
                self.chunks.append((row_index, 0, total_samples))
                continue
            stride = max(self.samples_per_frame, self.chunk_samples // 2) if training else self.chunk_samples
            start = 0
            while start < total_samples:
                valid = min(self.chunk_samples, total_samples - start)
                self.chunks.append((row_index, start, valid))
                if start + self.chunk_samples >= total_samples:
                    break
                start += stride
        if not self.chunks:
            raise RuntimeError(f"No Mimi feature chunks built from {manifest_path}")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, start_sample, valid_samples = self.chunks[index]
        row = self.rows[row_index]
        total_frames = self.chunk_samples // self.samples_per_frame
        valid_frames = min(total_frames, math.ceil(valid_samples / self.samples_per_frame))
        frame_valid_mask = torch.zeros(total_frames, dtype=torch.float32)
        frame_valid_mask[:valid_frames] = 1.0
        return {
            "id": row["id"],
            "session_id": row.get("session_id", row["id"]),
            "row_idx": row_index,
            "start_sample": start_sample,
            "valid_num_samples": valid_samples,
            "frame_valid_mask": frame_valid_mask,
        }


def collate_mimi_feature_chunks(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": [item["id"] for item in batch],
        "session_id": [item["session_id"] for item in batch],
        "row_idx": torch.tensor([item["row_idx"] for item in batch], dtype=torch.long),
        "start_sample": torch.tensor([item["start_sample"] for item in batch], dtype=torch.long),
        "valid_num_samples": torch.tensor([item["valid_num_samples"] for item in batch], dtype=torch.long),
        "frame_valid_mask": torch.stack([item["frame_valid_mask"] for item in batch], dim=0),
    }


def parse_float_list(value: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals or any(x <= 0 for x in vals):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive floats.")
    return vals


def vap_bin_times_to_frames(bin_times: list[float], frame_hz: float) -> list[int]:
    frames = [max(1, int(round(t * frame_hz))) for t in bin_times]
    if sum(frames) <= 0:
        raise ValueError(f"Invalid VAP bin frames from bin_times={bin_times}, frame_hz={frame_hz}")
    return frames


def model_frame_to_vad50_anchor(frame_idx: torch.Tensor, model_frame_hz: float) -> torch.Tensor:
    """Map model frame i to the first 50Hz VAD frame after that model frame."""
    return torch.round((frame_idx.to(torch.float32) + 1.0) * (50.0 / float(model_frame_hz))).to(torch.long)



def clean_vad_50hz(vad: torch.Tensor, *, min_speech_ms: int = 150, min_silence_ms: int = 150) -> torch.Tensor:
    """DualTurn official VAD cleanup: remove short speech and fill short silence."""
    out = vad.clone().to(torch.float32)
    n = int(out.numel())
    min_speech = int(min_speech_ms / 20.0)
    min_silence = int(min_silence_ms / 20.0)

    in_speech = False
    start = 0
    for i in range(n):
        if out[i] >= 0.5 and not in_speech:
            start = i
            in_speech = True
        elif out[i] < 0.5 and in_speech:
            if (i - start) < min_speech:
                out[start:i] = 0.0
            in_speech = False

    in_silence = False
    start = 0
    for i in range(n):
        if out[i] < 0.5 and not in_silence:
            start = i
            in_silence = True
        elif out[i] >= 0.5 and in_silence:
            if (i - start) < min_silence:
                out[start:i] = 1.0
            in_silence = False
    return out


def resample_vad_50hz(vad_50hz: torch.Tensor, target_frames: int, target_frame_hz: float) -> torch.Tensor:
    if target_frames <= 0:
        return vad_50hz.new_zeros((0,))
    if abs(target_frame_hz - 50.0) < 1e-6:
        out = vad_50hz[:target_frames]
        if out.numel() < target_frames:
            out = F.pad(out, (0, target_frames - out.numel()))
        return out

    out = vad_50hz.new_zeros((target_frames,))
    for i in range(target_frames):
        lo = int(math.floor(i * 50.0 / target_frame_hz))
        hi = int(math.floor((i + 1) * 50.0 / target_frame_hz))
        hi = max(hi, lo + 1)
        if lo < vad_50hz.numel():
            out[i] = 1.0 if vad_50hz[lo:min(hi, vad_50hz.numel())].float().mean() >= 0.5 else 0.0
    return out


class OfficialVadLabeler:
    """Build the shared 50 Hz binary VAD labels used by all FVAD targets.

    The public training path follows the paper pipeline: obtain per-channel
    Silero VAD, clean short speech/silence islands, then adapt that same VAD
    stream to the VAP and DualTurn frame grids.
    """

    def __init__(
        self,
        *,
        source: str,
        sample_rate: int,
        frame_hz: float,
        silero_threshold: float,
        silero_min_speech_ms: int,
        silero_min_silence_ms: int,
        clean_min_speech_ms: int,
        clean_min_silence_ms: int,
        cache_dir: Path | None = None,
    ) -> None:
        self.source = source
        self.sample_rate = int(sample_rate)
        self.frame_hz = float(frame_hz)
        self.silero_threshold = float(silero_threshold)
        self.silero_min_speech_ms = int(silero_min_speech_ms)
        self.silero_min_silence_ms = int(silero_min_silence_ms)
        self.clean_min_speech_ms = int(clean_min_speech_ms)
        self.clean_min_silence_ms = int(clean_min_silence_ms)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._cache: dict[str, torch.Tensor] = {}
        self.models = None
        if self.cache_dir is not None:
            print(f"VAD label source: cached Silero VAD at {self.cache_dir}")
            return
        if source == "silero":
            from silero_vad import load_silero_vad

            self.models = [load_silero_vad().eval(), load_silero_vad().eval()]
            print("VAD label source: Silero official DualTurn preprocessing")
        else:
            raise ValueError(f"Unsupported vad_source={source}; public training supports source='silero'")

    @torch.no_grad()

    def _load_cached_vad(self, session_id: str) -> torch.Tensor:
        if session_id not in self._cache:
            assert self.cache_dir is not None
            path = self.cache_dir / f"{session_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing cached VAD for {session_id}: {path}")
            import numpy as np

            with np.load(path) as data:
                arr = data["vad_50hz"].astype("float32")
            self._cache[session_id] = torch.from_numpy(arr)
        return self._cache[session_id]

    def _cached_batch(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        assert self.cache_dir is not None
        frame_valid_mask = batch["frame_valid_mask"].detach().cpu()
        target_T = int(frame_valid_mask.shape[1])
        target_50 = int(math.ceil(target_T * 50.0 / self.frame_hz)) + 128
        out = torch.zeros((len(batch["id"]), 2, target_50), dtype=torch.float32)
        for b, session_id in enumerate(batch["id"]):
            vad_50 = self._load_cached_vad(str(session_id))
            start_sample = int(batch["start_sample"][b].detach().cpu().item())
            start_50 = int(round(start_sample / float(self.sample_rate) * 50.0))
            chunk_50 = vad_50[:, start_50:start_50 + target_50]
            out[b, :, :chunk_50.shape[1]] = chunk_50
        return out.to(device)

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        if self.cache_dir is not None:
            return self._cached_batch(batch, device)
        from silero_vad import get_speech_timestamps

        audio = batch["audio"].detach().float().cpu()
        frame_valid_mask = batch["frame_valid_mask"].detach().cpu()
        B, C, _ = audio.shape
        target_T = int(frame_valid_mask.shape[1])
        target_50 = int(math.ceil(target_T * 50.0 / self.frame_hz))
        out = torch.zeros((B, 2, target_50), dtype=torch.float32)
        assert self.models is not None
        for b in range(B):
            for ch in range(min(C, 2)):
                wav = audio[b, ch]
                if self.sample_rate != 16_000:
                    wav = torchaudio.functional.resample(wav, self.sample_rate, 16_000)
                wav = wav.contiguous()
                timestamps = get_speech_timestamps(
                    wav,
                    self.models[ch],
                    sampling_rate=16_000,
                    threshold=self.silero_threshold,
                    min_speech_duration_ms=self.silero_min_speech_ms,
                    min_silence_duration_ms=self.silero_min_silence_ms,
                )
                n_50 = int(wav.numel() // 320)
                vad_50 = torch.zeros((n_50,), dtype=torch.float32)
                for ts in timestamps:
                    s = int(ts["start"] // 320)
                    e = min(int(ts["end"] // 320), n_50)
                    if s < e:
                        vad_50[s:e] = 1.0
                vad_50 = clean_vad_50hz(
                    vad_50,
                    min_speech_ms=self.clean_min_speech_ms,
                    min_silence_ms=self.clean_min_silence_ms,
                )
                out[b, ch, :min(target_50, vad_50.numel())] = vad_50[:target_50]
        return out.to(device)

def future_vad_states(
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    bin_frames_50hz: list[int],
    *,
    model_frame_hz: float,
    threshold_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eight future-VAD bits and a valid-frame mask."""
    if vad_50hz.dim() != 3 or vad_50hz.shape[1] != 2:
        raise ValueError(f"Expected 50Hz VAD [B, 2, T50], got {tuple(vad_50hz.shape)}")

    B, _, T50 = vad_50hz.shape
    T_model = int(frame_valid_mask.shape[1])
    states = torch.zeros((B, T_model, 8), dtype=torch.float32, device=vad_50hz.device)
    valid_frames = torch.zeros((B, T_model), dtype=torch.bool, device=vad_50hz.device)
    horizon = int(sum(bin_frames_50hz))
    frame_idx = torch.arange(T_model, device=vad_50hz.device)
    anchors_50 = model_frame_to_vad50_anchor(frame_idx, model_frame_hz)

    for b in range(B):
        valid = (frame_valid_mask[b].to(vad_50hz.device) > 0.5) & ((anchors_50 + horizon) <= T50)
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        valid_frames[b, valid_idx] = True
        bit = 0
        for ch in range(2):
            cs = torch.cat([
                vad_50hz.new_zeros((1,)),
                torch.cumsum(vad_50hz[b, ch].float(), dim=0),
            ])
            offset = 0
            for width in bin_frames_50hz:
                lo = anchors_50[valid_idx] + offset
                hi = lo + width
                mean = (cs[hi] - cs[lo]) / float(width)
                states[b, valid_idx, bit] = (mean >= threshold_ratio).float()
                offset += width
                bit += 1
    return states, valid_frames


def vad50_to_model_vad(
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    model_frame_hz: float,
    threshold_ratio: float,
) -> torch.Tensor:
    """Aggregate the shared 50Hz VAD onto the model frame grid."""
    B, C, T50 = vad_50hz.shape
    T_model = int(frame_valid_mask.shape[1])
    out = torch.zeros((B, C, T_model), dtype=torch.float32, device=vad_50hz.device)
    for i in range(T_model):
        lo = int(math.floor(i * 50.0 / model_frame_hz))
        hi = max(lo + 1, int(math.floor((i + 1) * 50.0 / model_frame_hz)))
        if lo < T50:
            out[:, :, i] = (
                vad_50hz[:, :, lo:min(hi, T50)].float().mean(dim=-1) >= threshold_ratio
            ).float()
    return out * frame_valid_mask[:, None, :].to(out.device)


def dualturn_native_fvad_targets(
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    threshold_ratio: float,
    bin_edges: list[int] = [3, 6, 12, 25],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official DualTurn soft occupancy targets on its native 12.5Hz grid."""
    vad = vad50_to_model_vad(
        vad_50hz,
        frame_valid_mask,
        model_frame_hz=12.5,
        threshold_ratio=threshold_ratio,
    )
    B, _, T = vad.shape
    states = torch.zeros((B, T, len(bin_edges) * 2), dtype=torch.float32, device=vad.device)
    valid = torch.zeros((B, T), dtype=torch.bool, device=vad.device)
    max_offset = int(bin_edges[-1])
    for b in range(B):
        valid_len = int((frame_valid_mask[b] > 0.5).sum().item())
        valid_T = valid_len - max_offset
        if valid_T <= 0:
            continue
        valid[b, :valid_T] = True
        idx = torch.arange(valid_T, device=vad.device)
        for ch in range(2):
            cs = torch.cat([vad.new_zeros((1,)), torch.cumsum(vad[b, ch], dim=0)])
            previous = 0
            for bin_idx, edge in enumerate(bin_edges):
                lo = idx + previous + 1
                hi = idx + int(edge) + 1
                states[b, :valid_T, ch * len(bin_edges) + bin_idx] = (
                    cs[hi] - cs[lo]
                ) / float(edge - previous)
                previous = int(edge)
    return states, valid


def fvad_targets(
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    bin_frames_50hz: list[int],
    *,
    model_frame_hz: float,
    threshold_ratio: float,
    target_scheme: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_scheme == "shared-binary":
        return future_vad_states(
            vad_50hz,
            frame_valid_mask,
            bin_frames_50hz,
            model_frame_hz=model_frame_hz,
            threshold_ratio=threshold_ratio,
        )
    if target_scheme == "native-soft":
        if abs(model_frame_hz - 12.5) > 1e-6:
            raise ValueError("native-soft FVAD targets are only defined for DualTurn at 12.5Hz")
        return dualturn_native_fvad_targets(
            vad_50hz,
            frame_valid_mask,
            threshold_ratio=threshold_ratio,
        )
    raise ValueError(f"Unsupported target_scheme={target_scheme}")


def vap256_labels(
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    bin_frames_50hz: list[int],
    *,
    model_frame_hz: float,
    threshold_ratio: float,
) -> torch.Tensor:
    """Pack the eight future-VAD bits into the original VAP 256-class label."""
    states, valid = future_vad_states(
        vad_50hz,
        frame_valid_mask,
        bin_frames_50hz,
        model_frame_hz=model_frame_hz,
        threshold_ratio=threshold_ratio,
    )
    powers = torch.tensor([1 << i for i in range(8)], dtype=torch.long, device=vad_50hz.device)
    labels = (states.long() * powers).sum(dim=-1)
    labels[~valid] = IGNORE_INDEX
    return labels


def future_vad_loss(
    logits: torch.Tensor,
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    bin_frames_50hz: list[int],
    *,
    model_frame_hz: float,
    threshold_ratio: float,
    context_frames: int,
    objective_type: str,
    target_scheme: str,
) -> dict[str, torch.Tensor]:
    states, valid = fvad_targets(
        vad_50hz,
        frame_valid_mask,
        bin_frames_50hz,
        model_frame_hz=model_frame_hz,
        threshold_ratio=threshold_ratio,
        target_scheme=target_scheme,
    )
    return future_vad_loss_from_targets(
        logits,
        states,
        valid,
        context_frames=context_frames,
        objective_type=objective_type,
    )


def future_vad_loss_from_targets(
    logits: torch.Tensor,
    states: torch.Tensor,
    valid: torch.Tensor,
    *,
    context_frames: int,
    objective_type: str,
) -> dict[str, torch.Tensor]:
    T = min(logits.shape[1], states.shape[1], valid.shape[1])
    logits = logits[:, :T]
    states = states[:, :T]
    valid = valid[:, :T]
    if context_frames > 0:
        valid[:, :context_frames] = False

    valid_frames = valid.sum().float()
    if valid_frames.item() == 0:
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "nll_sum": zero.detach(),
            "valid_frames": zero.detach(),
            "correct_frames": zero.detach(),
        }

    if objective_type == "categorical256":
        powers = torch.tensor([1 << i for i in range(8)], dtype=torch.long, device=logits.device)
        labels = (states.long() * powers).sum(dim=-1)
        nll_sum = F.cross_entropy(logits[valid], labels[valid], reduction="sum")
        pred = logits.argmax(dim=-1)
        correct = (pred[valid] == labels[valid]).sum().float()
    elif objective_type == "bernoulli8":
        if logits.shape[-1] != 8:
            raise ValueError(f"DualTurn FVAD expects 8 logits, got {logits.shape[-1]}")
        frame_nll = F.binary_cross_entropy_with_logits(logits, states, reduction="none").mean(dim=-1)
        nll_sum = frame_nll[valid].sum()
        pred_bits = logits >= 0
        correct = (pred_bits[valid] == states[valid].bool()).all(dim=-1).sum().float()
    else:
        raise ValueError(f"Unsupported objective_type={objective_type}")
    return {
        "loss": nll_sum / valid_frames,
        "nll_sum": nll_sum.detach(),
        "valid_frames": valid_frames.detach(),
        "correct_frames": correct.detach(),
    }


def dualturn_event_targets(
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    threshold_ratio: float,
) -> dict[str, torch.Tensor]:
    """Derive the official-style VAD/EOT/HOLD/BOT/BC pseudo-labels from shared VAD."""
    from turnnat.data.vad import derive_signals

    vad = vad50_to_model_vad(
        vad_50hz,
        frame_valid_mask,
        model_frame_hz=12.5,
        threshold_ratio=threshold_ratio,
    )
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in ["vad", "eot", "hold", "bot", "bc"]}
    for b in range(vad.shape[0]):
        signals = derive_signals(vad[b], frame_valid_mask[b], frame_hz=12.5)
        for name in rows:
            rows[name].append(signals[name].transpose(0, 1))
    return {name: torch.stack(values, dim=0).to(vad.device) for name, values in rows.items()}


def masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    mask = valid.unsqueeze(-1).expand_as(logits)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    logits_v = logits[mask]
    targets_v = targets[mask]
    bce = F.binary_cross_entropy_with_logits(logits_v, targets_v, reduction="none")
    probs = torch.sigmoid(logits_v)
    pt = targets_v * probs + (1.0 - targets_v) * (1.0 - probs)
    alpha_t = targets_v * alpha + (1.0 - targets_v) * (1.0 - alpha)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def vap_current_vad_loss(
    logits: torch.Tensor,
    vad_50hz: torch.Tensor,
    frame_valid_mask: torch.Tensor,
) -> torch.Tensor:
    T = min(logits.shape[1], frame_valid_mask.shape[1], vad_50hz.shape[-1])
    targets = vad_50hz[:, :, :T].transpose(1, 2).to(logits.device)
    valid = frame_valid_mask[:, :T].to(logits.device) > 0.5
    mask = valid.unsqueeze(-1).expand_as(logits[:, :T])
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[:, :T][mask], targets[mask])


def dualturn_multitask_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    frame_valid_mask: torch.Tensor,
    fvad_loss: torch.Tensor,
    *,
    weights: dict[str, float],
    alphas: dict[str, float],
    focal_gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    T = min(predictions["fvad"].shape[1], frame_valid_mask.shape[1])
    valid = frame_valid_mask[:, :T] > 0.5
    losses: dict[str, torch.Tensor] = {"fvad": fvad_loss}
    vad_logits = predictions["vad"][:, :T]
    vad_targets = targets["vad"][:, :T]
    mask = valid.unsqueeze(-1).expand_as(vad_logits)
    losses["vad"] = F.binary_cross_entropy_with_logits(vad_logits[mask], vad_targets[mask])
    for task in ["eot", "hold", "bot", "bc"]:
        losses[task] = masked_focal_loss(
            predictions[task][:, :T],
            targets[task][:, :T],
            valid,
            alpha=alphas[task],
            gamma=focal_gamma,
        )
    active_weight = sum(float(weights[name]) for name in losses if float(weights[name]) > 0)
    if active_weight <= 0:
        raise ValueError("At least one DualTurn task loss weight must be positive")
    # Match DualTurn's official combined_loss: task weights scale a sum rather
    # than a normalized weighted mean.
    total = sum(float(weights[name]) * loss for name, loss in losses.items())
    return total, losses


def configure_cfg_for_backbone(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("paths", {})
    cfg.setdefault("data", {})
    cfg.setdefault("precision", {})

    if args.output_dir:
        cfg["paths"]["output_dir"] = str(args.output_dir)
    else:
        cfg["paths"]["output_dir"] = str(Path(cfg["paths"]["output_dir"]) / f"vap256_{args.backbone}_{args.train_mode}")

    if args.train_manifest is not None:
        cfg["data"]["train_manifest"] = str(args.train_manifest)
    if args.val_manifest is not None:
        cfg["data"]["val_manifest"] = str(args.val_manifest)
    if getattr(args, "test1_manifest", None) is not None:
        cfg["data"]["test1_manifest"] = str(args.test1_manifest)

    if args.backbone == "vap":
        cfg["data"]["target_sample_rate"] = 16_000
        cfg["data"]["samples_per_frame"] = 320
        cfg["data"]["frame_hz"] = 50.0
    else:
        cfg["data"]["target_sample_rate"] = 24_000
        cfg["data"]["samples_per_frame"] = 1920
        cfg["data"]["frame_hz"] = 12.5

    if args.chunk_seconds is not None:
        cfg["data"]["chunk_seconds"] = float(args.chunk_seconds)
        cfg["data"]["eval_chunk_seconds"] = float(args.chunk_seconds)
    if args.train_batch_size is not None:
        cfg["data"]["train_batch_size"] = int(args.train_batch_size)
    if args.eval_batch_size is not None:
        cfg["data"]["eval_batch_size"] = int(args.eval_batch_size)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    if args.precision is not None:
        cfg["precision"]["mode"] = args.precision

    ensure_output_dirs(cfg)
    return cfg


def set_trainable(model: nn.Module, backbone: str, train_mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for p in model.head_parameters():
        p.requires_grad = True

    if train_mode == "head":
        return

    if backbone == "vap":
        if train_mode == "adapter":
            for name, p in model.backbone.named_parameters():
                if name.startswith("transformer.") or name.startswith("feature_projection."):
                    p.requires_grad = True
        elif train_mode == "full":
            for p in model.backbone.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unsupported train_mode={train_mode}")
        return

    if train_mode == "adapter":
        for name, p in model.named_parameters():
            if "mimi_projection" in name or "lora_" in name:
                p.requires_grad = True
        if model.task_layer_weights is not None:
            tasks = model.TASKS if model.multitask else ("fvad",)
            for task in tasks:
                model.task_layer_weights[task].requires_grad = True
    elif train_mode == "full":
        for p in model.model.backbone.parameters():
            p.requires_grad = True
        for p in model.model.mimi_projection.parameters():
            p.requires_grad = True
        if model.task_layer_weights is not None:
            tasks = model.TASKS if model.multitask else ("fvad",)
            for task in tasks:
                model.task_layer_weights[task].requires_grad = True
    else:
        raise ValueError(f"Unsupported train_mode={train_mode}")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


def precision_policy(mode: str, device: torch.device) -> tuple[bool, torch.dtype | None, bool]:
    if device.type != "cuda":
        return False, None, False
    mode = str(mode).lower()
    if mode in {"bf16", "bfloat16"}:
        return True, torch.bfloat16, False
    if mode in {"fp16", "float16", "half"}:
        return True, torch.float16, True
    if mode in {"fp32", "float32", "32"}:
        return False, None, False
    raise ValueError(f"Unsupported precision mode: {mode}")



def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def run_epoch(
    *,
    model: nn.Module,
    vad_labeler: OfficialVadLabeler,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    use_autocast: bool,
    autocast_dtype: torch.dtype | None,
    bin_frames: list[int],
    model_frame_hz: float,
    threshold_ratio: float,
    context_frames: int,
    grad_clip: float | None,
    max_batches: int | None,
    desc: str,
    wandb_run: Any | None = None,
    wandb_prefix: str | None = None,
    wandb_log_every: int = 0,
    epoch: int | None = None,
    global_step_state: dict[str, int] | None = None,
    eval_every_steps: int = 0,
    eval_loaders: dict[str, DataLoader] | None = None,
    eval_max_batches: int | None = None,
    eval_history: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "nll_sum": 0.0,
        "valid_frames": 0.0,
        "correct_frames": 0.0,
        "objective_loss_sum": 0.0,
        "num_batches": 0.0,
    }
    task_loss_sums: dict[str, float] = {}
    pbar = tqdm(loader, desc=desc, leave=False)
    for step, batch in enumerate(pbar, start=1):
        if max_batches is not None and step > max_batches:
            break

        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=use_autocast, dtype=autocast_dtype):
                predictions = None
                mimi_features = None
                feature_cache = getattr(model, "mimi_feature_cache", None)
                if feature_cache is not None:
                    feat0, feat1 = feature_cache.batch(batch, device)
                    mimi_features = {"mimi_feat_ch0": feat0, "mimi_feat_ch1": feat1}
                if getattr(model, "vap_vad_loss_weight", 0.0) > 0:
                    predictions = model.forward_all(batch["audio"])
                    logits = predictions["fvad"]
                elif getattr(model, "multitask", False):
                    predictions = model.forward_all(
                        None if mimi_features is not None else batch["audio"],
                        **(mimi_features or {}),
                    )
                    logits = predictions["fvad"]
                elif mimi_features is not None:
                    logits = model.forward_fvad(**mimi_features)
                else:
                    logits = model(
                        audio=batch["audio"],
                        sample_mask=batch["sample_mask"],
                        frame_valid_mask=batch["frame_valid_mask"],
                    )
                vad_50hz = vad_labeler(batch, device)
                signal_cache = getattr(model, "signal_label_cache", None)
                cached_signals = (
                    signal_cache.batch(batch, device)
                    if predictions is not None and getattr(model, "multitask", False) and signal_cache is not None
                    else None
                )
                use_cached_native_fvad = (
                    cached_signals is not None
                    and model.objective_type == "bernoulli8"
                    and model.target_scheme == "native-soft"
                )
                if use_cached_native_fvad:
                    cached_valid = (cached_signals["fvad_mask"] > 0.5) & (
                        batch["frame_valid_mask"] > 0.5
                    )
                    out = future_vad_loss_from_targets(
                        logits,
                        cached_signals["fvad"],
                        cached_valid,
                        context_frames=context_frames,
                        objective_type=model.objective_type,
                    )
                else:
                    out = future_vad_loss(
                        logits,
                        vad_50hz,
                        batch["frame_valid_mask"],
                        bin_frames,
                        model_frame_hz=model_frame_hz,
                        threshold_ratio=threshold_ratio,
                        context_frames=context_frames,
                        objective_type=model.objective_type,
                        target_scheme=model.target_scheme,
                    )
                loss = out["loss"]
                if predictions is not None and getattr(model, "vap_vad_loss_weight", 0.0) > 0:
                    vad_aux_loss = vap_current_vad_loss(
                        predictions["vad"],
                        vad_50hz,
                        batch["frame_valid_mask"],
                    )
                    loss = loss + float(model.vap_vad_loss_weight) * vad_aux_loss
                    out["task_losses"] = {
                        "fvad": out["loss"].detach(),
                        "vad": vad_aux_loss.detach(),
                    }
                elif predictions is not None:
                    event_targets = cached_signals
                    if event_targets is None:
                        raise RuntimeError("All-six DualTurn training requires cached official signal labels")
                    loss, task_losses = dualturn_multitask_loss(
                        predictions,
                        event_targets,
                        batch["frame_valid_mask"],
                        loss,
                        weights=model.multitask_weights,
                        alphas=model.event_alphas,
                        focal_gamma=model.event_focal_gamma,
                    )
                    out["task_losses"] = {k: v.detach() for k, v in task_losses.items()}

        if training:
            assert scaler is not None
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        totals["nll_sum"] += float(out["nll_sum"].cpu())
        totals["valid_frames"] += float(out["valid_frames"].cpu())
        totals["correct_frames"] += float(out["correct_frames"].cpu())
        totals["objective_loss_sum"] += float(loss.detach().cpu())
        totals["num_batches"] += 1.0
        for task_name, task_loss in out.get("task_losses", {}).items():
            task_loss_sums[task_name] = task_loss_sums.get(task_name, 0.0) + float(task_loss.cpu())

        global_step = None
        if training and global_step_state is not None:
            global_step_state["step"] = int(global_step_state.get("step", 0)) + 1
            global_step = global_step_state["step"]

        if totals["valid_frames"] > 0:
            running_fvad_nll = totals["nll_sum"] / totals["valid_frames"]
            running_objective_loss = totals["objective_loss_sum"] / totals["num_batches"]
            running_acc = totals["correct_frames"] / totals["valid_frames"]
            pbar.set_postfix(
                loss=f"{running_objective_loss:.4f}",
                fvad_nll=f"{running_fvad_nll:.4f}",
                acc=f"{running_acc:.4f}",
            )
            if (
                training
                and wandb_run is not None
                and wandb_prefix is not None
                and wandb_log_every > 0
                and (step == 1 or step % wandb_log_every == 0)
            ):
                log_payload = {
                    f"{wandb_prefix}/batch_loss": float(loss.detach().cpu()),
                    f"{wandb_prefix}/running_loss": running_objective_loss,
                    f"{wandb_prefix}/running_objective_loss": running_objective_loss,
                    f"{wandb_prefix}/running_fvad_nll": running_fvad_nll,
                    f"{wandb_prefix}/running_acc": running_acc,
                    f"{wandb_prefix}/valid_frames_seen": totals["valid_frames"],
                    f"{wandb_prefix}/batch": step,
                    "epoch": epoch,
                }
                if global_step is not None:
                    log_payload["global_step"] = global_step
                for task_name, task_loss in out.get("task_losses", {}).items():
                    log_payload[f"{wandb_prefix}/task_{task_name}_loss"] = float(task_loss.cpu())
                wandb_run.log(log_payload, step=global_step)

        do_loss_eval = (
            training
            and global_step is not None
            and eval_every_steps > 0
            and eval_loaders
            and global_step % eval_every_steps == 0
        )
        if do_loss_eval:
            print(f"\n[future-VAD] mid-epoch eval at global_step={global_step}")
            eval_row: dict[str, Any] = {"epoch": epoch, "global_step": global_step, "eval": {}}
            if do_loss_eval and eval_loaders:
                for eval_name, eval_loader in eval_loaders.items():
                    with torch.no_grad():
                        metrics = run_epoch(
                            model=model,
                            vad_labeler=vad_labeler,
                            loader=eval_loader,
                            device=device,
                            optimizer=None,
                            scaler=None,
                            use_autocast=use_autocast,
                            autocast_dtype=autocast_dtype,
                            bin_frames=bin_frames,
                            model_frame_hz=model_frame_hz,
                            threshold_ratio=threshold_ratio,
                            context_frames=context_frames,
                            grad_clip=None,
                            max_batches=eval_max_batches,
                            desc=f"{eval_name} step{global_step}",
                        )
                    eval_row["eval"][eval_name] = metrics
                    print(
                        f"{eval_name}_loss={metrics['loss']:.6f} "
                        f"{eval_name}_fvad_nll={metrics['fvad_nll']:.6f} "
                        f"{eval_name}_acc={metrics['acc']:.4f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                f"{eval_name}/loss": metrics["loss"],
                                f"{eval_name}/fvad_nll": metrics["fvad_nll"],
                                f"{eval_name}/acc": metrics["acc"],
                                f"{eval_name}/nll_sum": metrics["nll_sum"],
                                f"{eval_name}/valid_frames": metrics["valid_frames"],
                                "epoch": epoch,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )
            if eval_history is not None:
                eval_history.append(eval_row)
            model.train(True)

    fvad_nll = totals["nll_sum"] / max(totals["valid_frames"], 1.0)
    objective_loss = totals["objective_loss_sum"] / max(totals["num_batches"], 1.0)
    acc = totals["correct_frames"] / max(totals["valid_frames"], 1.0)
    metrics = {"loss": objective_loss, "fvad_nll": fvad_nll, "acc": acc, **totals}
    metrics.update({
        f"task_{name}_loss": value / max(totals["num_batches"], 1.0)
        for name, value in task_loss_sums.items()
    })
    return metrics


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted local checkpoint including optimizer and run metadata."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch before weights_only was introduced.
        return torch.load(str(path), map_location="cpu")


def build_model(cfg: dict[str, Any], args: argparse.Namespace) -> nn.Module:
    dropout = float(args.head_dropout if args.head_dropout is not None else cfg.get("model", {}).get("head_dropout", 0.1))
    if args.backbone == "vap":
        model = VAP256Model(args.vap_ckpt, args.head_hidden_dim, dropout, args.head_init)
        model.target_scheme = "shared-binary"
        model.vap_vad_loss_weight = float(args.vap_vad_loss_weight)
        return model

    if args.head_hidden_dim > 0:
        raise ValueError("Official DualTurn uses its pretrained linear FVAD head; use --head-hidden-dim 0")
    target_scheme = args.fvad_target_scheme
    if target_scheme == "auto":
        target_scheme = "native-soft"
    multitask = args.dualturn_losses == "all"
    model = OfficialDualTurnFVADModel(
        model_id=args.dualturn_model_id,
        local_files_only=args.local_files_only,
        head_init=args.head_init,
        fvad_head_type=args.dualturn_fvad_head,
        multitask=multitask,
        use_lora=args.train_mode == "adapter",
        lora_r=args.dualturn_lora_r,
        lora_alpha=args.dualturn_lora_alpha,
        lora_dropout=args.dualturn_lora_dropout,
        lora_target_modules=args.dualturn_lora_targets,
    )
    if args.dualturn_fvad_head == "categorical256" and target_scheme != "shared-binary":
        raise ValueError("DualTurn categorical256 requires --fvad-target-scheme shared-binary")
    model.target_scheme = target_scheme
    model.multitask_weights = {
        "fvad": args.weight_fvad,
        "vad": args.weight_vad,
        "eot": args.weight_eot,
        "hold": args.weight_hold,
        "bot": args.weight_bot,
        "bc": args.weight_bc,
    }
    model.event_alphas = {
        "eot": args.eot_alpha,
        "hold": args.hold_alpha,
        "bot": args.bot_alpha,
        "bc": args.bc_alpha,
    }
    model.event_focal_gamma = args.event_focal_gamma
    model.mimi_feature_cache = (
        MimiFeatureCache(args.mimi_feature_root)
        if args.mimi_feature_root is not None
        else None
    )
    model.signal_label_cache = (
        SignalLabelCache(args.signal_label_cache_dir)
        if args.signal_label_cache_dir is not None
        else None
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fine-tune the native future-VAD head of a VAP or official DualTurn backbone."
        )
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", choices=["vap", "dualturn"], required=True)
    ap.add_argument("--train-mode", choices=["head", "adapter", "full"], default="head")
    ap.add_argument("--vap-ckpt", type=Path, default=DEFAULT_VAP_CKPT)
    ap.add_argument("--dualturn-model-id", default=DEFAULT_DUALTURN_MODEL_ID)
    ap.add_argument(
        "--dualturn-fvad-head",
        choices=["native8", "categorical256"],
        default="native8",
        help="Use the official 8-logit FVAD head or a new VAP-compatible 256-class head.",
    )
    ap.add_argument("--mimi-feature-root", type=Path, default=None)
    ap.add_argument("--signal-label-cache-dir", type=Path, default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--resume-ckpt", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--train-manifest", type=Path, default=None)
    ap.add_argument("--val-manifest", type=Path, default=None)
    ap.add_argument("--test1-manifest", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--fvad-head-lr",
        type=float,
        default=None,
        help="Optional FVAD-head LR, useful for a randomly initialized DualTurn 256-class head.",
    )
    ap.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Optional separate LR for non-head trainable parameters in adapter/full modes.",
    )
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--chunk-seconds", type=float, default=None)
    ap.add_argument("--train-batch-size", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--head-hidden-dim", type=int, default=0)
    ap.add_argument("--head-dropout", type=float, default=None)
    ap.add_argument("--head-init", choices=["auto", "pretrained", "random"], default="auto")
    ap.add_argument(
        "--fvad-target-scheme",
        choices=["auto", "shared-binary", "native-soft"],
        default="auto",
    )
    ap.add_argument("--dualturn-losses", choices=["fvad", "all"], default="fvad")
    ap.add_argument("--dualturn-lora-r", type=int, default=16)
    ap.add_argument("--dualturn-lora-alpha", type=int, default=32)
    ap.add_argument("--dualturn-lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--dualturn-lora-targets",
        type=lambda value: [x.strip() for x in value.split(",") if x.strip()],
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    ap.add_argument("--weight-fvad", type=float, default=1.0)
    ap.add_argument(
        "--vap-vad-loss-weight",
        type=float,
        default=0.0,
        help="Current-VAD BCE weight; use 1.0 for the official VAP full-training loss.",
    )
    ap.add_argument("--weight-vad", type=float, default=1.0)
    ap.add_argument("--weight-eot", type=float, default=1.0)
    ap.add_argument("--weight-hold", type=float, default=1.0)
    ap.add_argument("--weight-bot", type=float, default=1.0)
    ap.add_argument("--weight-bc", type=float, default=1.0)
    ap.add_argument("--eot-alpha", type=float, default=0.75)
    ap.add_argument("--hold-alpha", type=float, default=0.60)
    ap.add_argument("--bot-alpha", type=float, default=0.80)
    ap.add_argument("--bc-alpha", type=float, default=0.80)
    ap.add_argument("--event-focal-gamma", type=float, default=2.0)
    ap.add_argument("--vap-bin-times", type=parse_float_list, default=DEFAULT_VAP_BIN_TIMES)
    ap.add_argument("--threshold-ratio", type=float, default=0.5)
    ap.add_argument("--context-seconds", type=float, default=0.0)
    ap.add_argument("--vad-source", choices=["silero"], default="silero")
    ap.add_argument("--vad-cache-dir", type=Path, default=None)
    ap.add_argument("--silero-threshold", type=float, default=0.5)
    ap.add_argument("--silero-min-speech-ms", type=int, default=100)
    ap.add_argument("--silero-min-silence-ms", type=int, default=50)
    ap.add_argument("--clean-min-speech-ms", type=int, default=150)
    ap.add_argument("--clean-min-silence-ms", type=int, default=150)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--max-mid-eval-batches", type=int, default=None)
    ap.add_argument("--eval-every-steps", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--wandb-log-every", type=int, default=20)
    args = ap.parse_args()

    if args.backbone == "dualturn" and args.dualturn_losses == "all" and args.signal_label_cache_dir is None:
        raise ValueError(
            "DualTurn all-six training requires --signal-label-cache-dir generated by "
            "build_dualturn_signal_cache.py"
        )

    cfg = configure_cfg_for_backbone(load_config(args.config), args)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision(str(cfg.get("precision", {}).get("matmul_precision", "high")))

    using_mimi_features = args.backbone == "dualturn" and args.mimi_feature_root is not None
    if using_mimi_features and args.vad_cache_dir is None:
        raise ValueError("--mimi-feature-root requires --vad-cache-dir so training never falls back to WAV VAD")
    dataset_cls = MimiFeatureChunkDataset if using_mimi_features else StereoChunkDataset
    collate_fn = collate_mimi_feature_chunks if using_mimi_features else collate_chunks
    train_ds = dataset_cls(cfg["data"]["train_manifest"], cfg, training=True)
    val_ds = dataset_cls(cfg["data"]["val_manifest"], cfg, training=False)
    test1_ds = None
    if args.test1_manifest is not None:
        test1_ds = dataset_cls(str(args.test1_manifest), cfg, training=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["train_batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=bool(cfg["data"].get("pin_memory", False)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=bool(cfg["data"].get("pin_memory", False)),
        collate_fn=collate_fn,
    )
    test1_loader = None
    if test1_ds is not None:
        test1_loader = DataLoader(
            test1_ds,
            batch_size=int(cfg["data"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 0)),
            pin_memory=bool(cfg["data"].get("pin_memory", False)),
            collate_fn=collate_fn,
        )

    frame_hz = float(cfg["data"]["frame_hz"])
    bin_frames = vap_bin_times_to_frames(args.vap_bin_times, 50.0)
    context_frames = int(math.ceil(args.context_seconds * frame_hz))

    vad_labeler = OfficialVadLabeler(
        source=args.vad_source,
        sample_rate=int(cfg["data"]["target_sample_rate"]),
        frame_hz=frame_hz,
        silero_threshold=args.silero_threshold,
        silero_min_speech_ms=args.silero_min_speech_ms,
        silero_min_silence_ms=args.silero_min_silence_ms,
        clean_min_speech_ms=args.clean_min_speech_ms,
        clean_min_silence_ms=args.clean_min_silence_ms,
        cache_dir=args.vad_cache_dir,
    )

    model = build_model(cfg, args).to(device)
    model.checkpoint_args = vars(args).copy()
    start_epoch = 1
    initial_global_step = 0
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    resume_payload: dict[str, Any] | None = None

    if args.resume_ckpt is not None:
        resume_payload = load_training_checkpoint(args.resume_ckpt)
        model.load_state_dict(resume_payload["model_state"], strict=True)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        history = list(resume_payload.get("history", []))
        prior_val_losses = [
            float(row["val"]["loss"])
            for row in history
            if isinstance(row.get("val"), dict) and row["val"].get("loss") is not None
        ]
        best_val = min([float(resume_payload.get("best_val", best_val)), *prior_val_losses])
        initial_global_step = int(resume_payload.get("global_step", 0))
        if initial_global_step <= 0 and history:
            initial_global_step = int(history[-1].get("global_step", 0))
        print(
            f"Resumed from {args.resume_ckpt}; start_epoch={start_epoch} "
            f"global_step={initial_global_step} best_val={best_val:.6f}"
        )

    set_trainable(model, args.backbone, args.train_mode)
    trainable, total = count_trainable_parameters(model)
    print(f"Backbone={args.backbone} train_mode={args.train_mode}")
    print(
        f"FVAD target={model.target_scheme} objective={model.objective_type} "
        f"losses={'all-six' if getattr(model, 'multitask', False) else 'fvad-only'}"
    )
    print(f"Frame Hz={frame_hz}, label Hz=50.0, VAP bin_times={args.vap_bin_times}, bin_frames_50hz={bin_frames}")
    print(f"Trainable params={trainable:,} / total={total:,}")

    wandb_run = None
    if args.wandb_project is not None and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("wandb is not installed; install it or omit --wandb-project") from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "backbone": args.backbone,
                "train_mode": args.train_mode,
                "head_init": args.head_init,
                "objective_type": model.objective_type,
                "dualturn_fvad_head": args.dualturn_fvad_head,
                "train_manifest": str(cfg["data"]["train_manifest"]),
                "val_manifest": str(cfg["data"]["val_manifest"]),
                "test1_manifest": str(args.test1_manifest) if args.test1_manifest is not None else None,
                "output_dir": str(cfg["paths"]["output_dir"]),
                "epochs": args.epochs,
                "lr": args.lr,
                "fvad_head_lr": args.fvad_head_lr,
                "backbone_lr": args.backbone_lr,
                "fvad_target_scheme": model.target_scheme,
                "dualturn_losses": args.dualturn_losses,
                "vap_vad_loss_weight": args.vap_vad_loss_weight,
                "mimi_feature_root": str(args.mimi_feature_root) if args.mimi_feature_root else None,
                "signal_label_cache_dir": str(args.signal_label_cache_dir) if args.signal_label_cache_dir else None,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "chunk_seconds": cfg["data"].get("chunk_seconds"),
                "train_batch_size": cfg["data"]["train_batch_size"],
                "eval_batch_size": cfg["data"]["eval_batch_size"],
                "frame_hz": frame_hz,
                "vap_bin_times": args.vap_bin_times,
                "vap_bin_frames_50hz": bin_frames,
                "label_frame_hz": 50.0,
                "threshold_ratio": args.threshold_ratio,
                "context_seconds": args.context_seconds,
                "vad_source": args.vad_source,
                "vad_cache_dir": str(args.vad_cache_dir) if args.vad_cache_dir is not None else None,
                "trainable_params": trainable,
                "total_params": total,
                "wandb_log_every": args.wandb_log_every,
                "eval_every_steps": args.eval_every_steps,
                "max_mid_eval_batches": args.max_mid_eval_batches,
            },
        )

    head_ids = {id(p) for p in model.head_parameters()}
    fvad_head_ids = {id(p) for p in model.fvad_head_parameters()}
    fvad_head_params = [p for p in model.parameters() if p.requires_grad and id(p) in fvad_head_ids]
    other_head_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) in head_ids and id(p) not in fvad_head_ids
    ]
    shared_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    parameter_groups: list[dict[str, Any]] = []
    if fvad_head_params:
        parameter_groups.append({
            "params": fvad_head_params,
            "lr": float(args.fvad_head_lr if args.fvad_head_lr is not None else args.lr),
        })
    if other_head_params:
        parameter_groups.append({"params": other_head_params, "lr": float(args.lr)})
    if shared_params:
        parameter_groups.append({
            "params": shared_params,
            "lr": float(args.backbone_lr if args.backbone_lr is not None else args.lr),
        })
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(args.weight_decay))

    use_amp, amp_dtype, use_scaler = precision_policy(str(cfg["precision"].get("mode", "bf16")), device)
    scaler = GradScaler(device.type, enabled=use_scaler)


    if resume_payload is not None and "optimizer_state" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state"])

    ckpt_dir = Path(cfg["paths"]["output_dir"]) / "checkpoints"
    save_json(
        Path(cfg["paths"]["output_dir"]) / "artifacts" / "vap256_train_config.json",
        {
            "backbone": args.backbone,
            "train_mode": args.train_mode,
            "head_init": args.head_init,
            "objective_type": model.objective_type,
            "dualturn_fvad_head": args.dualturn_fvad_head,
            "fvad_target_scheme": model.target_scheme,
            "dualturn_losses": args.dualturn_losses,
            "vap_vad_loss_weight": args.vap_vad_loss_weight,
            "lr": args.lr,
            "fvad_head_lr": args.fvad_head_lr,
            "backbone_lr": args.backbone_lr,
            "mimi_feature_root": str(args.mimi_feature_root) if args.mimi_feature_root else None,
            "signal_label_cache_dir": str(args.signal_label_cache_dir) if args.signal_label_cache_dir else None,
            "dualturn_lora": {
                "r": args.dualturn_lora_r,
                "alpha": args.dualturn_lora_alpha,
                "dropout": args.dualturn_lora_dropout,
                "targets": args.dualturn_lora_targets,
            },
            "task_weights": {
                "fvad": args.weight_fvad,
                "vad": args.weight_vad,
                "eot": args.weight_eot,
                "hold": args.weight_hold,
                "bot": args.weight_bot,
                "bc": args.weight_bc,
            },
            "dualturn_model_id": args.dualturn_model_id,
            "local_files_only": args.local_files_only,
            "test1_manifest": str(args.test1_manifest) if args.test1_manifest is not None else None,
            "eval_every_steps": args.eval_every_steps,
            "max_mid_eval_batches": args.max_mid_eval_batches,
            "frame_hz": frame_hz,
            "vap_bin_times": args.vap_bin_times,
            "vap_bin_frames_50hz": bin_frames,
            "label_frame_hz": 50.0,
            "threshold_ratio": args.threshold_ratio,
            "context_seconds": args.context_seconds,
            "vad_source": args.vad_source,
            "vad_cache_dir": str(args.vad_cache_dir) if args.vad_cache_dir is not None else None,
            "silero_threshold": args.silero_threshold,
            "silero_min_speech_ms": args.silero_min_speech_ms,
            "silero_min_silence_ms": args.silero_min_silence_ms,
            "clean_min_speech_ms": args.clean_min_speech_ms,
            "clean_min_silence_ms": args.clean_min_silence_ms,
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            "wandb_mode": args.wandb_mode,
            "wandb_log_every": args.wandb_log_every,
            "label_definition": (
                "shared-binary uses the common 50Hz VAP windows and eight thresholded bits; "
                "native-soft downsamples the same 50Hz VAD to 12.5Hz and reproduces the "
                "official DualTurn [3,6,12,25] soft occupancy targets."
            ),
            "config": cfg,
        },
    )

    global_step_state = {"step": initial_global_step}
    prior_history_path = Path(cfg["paths"]["output_dir"]) / "artifacts" / "history.json"
    prior_artifacts: dict[str, Any] = {}
    if resume_payload is not None and prior_history_path.is_file():
        prior_artifacts = json.loads(prior_history_path.read_text(encoding="utf-8"))
    mid_eval_history: list[dict[str, Any]] = list(prior_artifacts.get("mid_eval_history", []))
    eval_loaders = {"dev": val_loader}
    if test1_loader is not None:
        eval_loaders["test1"] = test1_loader


    for epoch in range(start_epoch, int(args.epochs) + 1):
        print(f"\n[future-VAD] epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model=model,
            vad_labeler=vad_labeler,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_autocast=use_amp,
            autocast_dtype=amp_dtype,
            bin_frames=bin_frames,
            model_frame_hz=frame_hz,
            threshold_ratio=args.threshold_ratio,
            context_frames=context_frames,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
            desc=f"train e{epoch}",
            wandb_run=wandb_run,
            wandb_prefix="train",
            wandb_log_every=args.wandb_log_every,
            epoch=epoch,
            global_step_state=global_step_state,
            eval_every_steps=args.eval_every_steps,
            eval_loaders=eval_loaders,
            eval_max_batches=args.max_mid_eval_batches,
            eval_history=mid_eval_history,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                vad_labeler=vad_labeler,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=None,
                use_autocast=use_amp,
                autocast_dtype=amp_dtype,
                bin_frames=bin_frames,
                model_frame_hz=frame_hz,
                threshold_ratio=args.threshold_ratio,
                context_frames=context_frames,
                grad_clip=None,
                max_batches=args.max_val_batches,
                desc=f"val e{epoch}",
                wandb_run=None,
                wandb_prefix="val",
                wandb_log_every=0,
                epoch=epoch,
            )

        test1_metrics = None
        if test1_loader is not None:
            with torch.no_grad():
                test1_metrics = run_epoch(
                    model=model,
                    vad_labeler=vad_labeler,
                    loader=test1_loader,
                    device=device,
                    optimizer=None,
                    scaler=None,
                    use_autocast=use_amp,
                    autocast_dtype=amp_dtype,
                    bin_frames=bin_frames,
                    model_frame_hz=frame_hz,
                    threshold_ratio=args.threshold_ratio,
                    context_frames=context_frames,
                    grad_clip=None,
                    max_batches=args.max_val_batches,
                    desc=f"test1 e{epoch}",
                )

        row = {"epoch": epoch, "global_step": global_step_state["step"], "train": train_metrics, "val": val_metrics}
        if test1_metrics is not None:
            row["test1"] = test1_metrics
        history.append(row)
        msg = (
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_fvad_nll={train_metrics['fvad_nll']:.6f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_fvad_nll={val_metrics['fvad_nll']:.6f} "
            f"val_acc={val_metrics['acc']:.4f}"
        )
        if test1_metrics is not None:
            msg += (
                f" test1_loss={test1_metrics['loss']:.6f} "
                f"test1_fvad_nll={test1_metrics['fvad_nll']:.6f} "
                f"test1_acc={test1_metrics['acc']:.4f}"
            )
        print(msg)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/fvad_nll": train_metrics["fvad_nll"],
                    "train/acc": train_metrics["acc"],
                    "train/nll_sum": train_metrics["nll_sum"],
                    "train/valid_frames": train_metrics["valid_frames"],
                    "val/loss": val_metrics["loss"],
                    "val/fvad_nll": val_metrics["fvad_nll"],
                    "val/acc": val_metrics["acc"],
                    "val/nll_sum": val_metrics["nll_sum"],
                    "val/valid_frames": val_metrics["valid_frames"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "best_val_loss": min(best_val, float(val_metrics["loss"])),
                    "global_step": global_step_state["step"],
                },
                step=global_step_state["step"],
            )
            if test1_metrics is not None:
                wandb_run.log(
                    {
                        "test1/loss": test1_metrics["loss"],
                        "test1/fvad_nll": test1_metrics["fvad_nll"],
                        "test1/acc": test1_metrics["acc"],
                        "test1/nll_sum": test1_metrics["nll_sum"],
                        "test1/valid_frames": test1_metrics["valid_frames"],
                        "epoch": epoch,
                        "global_step": global_step_state["step"],
                    },
                    step=global_step_state["step"],
                )

        improved = float(val_metrics["loss"]) < best_val
        if improved:
            best_val = float(val_metrics["loss"])
        payload = {
            "epoch": epoch,
            "global_step": global_step_state["step"],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
            "history": history,
            "args": vars(args),
        }
        save_checkpoint(ckpt_dir / "last.pt", payload)
        if improved:
            save_checkpoint(ckpt_dir / "best.pt", payload)
            print(f"Saved best checkpoint: {ckpt_dir / 'best.pt'}")

    save_json(
        Path(cfg["paths"]["output_dir"]) / "artifacts" / "history.json",
        {"history": history, "mid_eval_history": mid_eval_history},
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
