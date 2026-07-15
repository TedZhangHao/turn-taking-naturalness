from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from dualturn.models.backbone import MimiQwenBackbone


class SparseHead(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DenseHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Stage2DualTurnModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = MimiQwenBackbone(cfg)
        hidden_size = int(self.backbone.qwen.config.hidden_size)
        dropout = float(cfg["model"].get("head_dropout", 0.1))

        self.heads = nn.ModuleDict()
        for prefix in ["u", "a"]:
            for name in ["eot", "hold", "bot", "bc"]:
                self.heads[f"{prefix}_{name}"] = SparseHead(hidden_size, dropout)
            self.heads[f"{prefix}_vad"] = DenseHead(hidden_size, 1)
            self.heads[f"{prefix}_fvad"] = DenseHead(hidden_size, 4)

    def forward(
        self,
        audio: torch.Tensor,
        sample_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(audio=audio, sample_mask=sample_mask, frame_valid_mask=frame_valid_mask)
        hidden = out.hidden

        pred: dict[str, torch.Tensor] = {}
        for prefix in ["u", "a"]:
            for name in ["eot", "hold", "bot", "bc"]:
                pred[f"{prefix}_{name}"] = self.heads[f"{prefix}_{name}"](hidden)
            pred[f"{prefix}_vad"] = self.heads[f"{prefix}_vad"](hidden).squeeze(-1)
            pred[f"{prefix}_fvad"] = self.heads[f"{prefix}_fvad"](hidden)

        pred["frame_valid_mask"] = frame_valid_mask[:, : hidden.shape[1]]
        return pred

    def predict_signal_probs(
        self,
        audio: torch.Tensor,
        sample_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = self.forward(audio=audio, sample_mask=sample_mask, frame_valid_mask=frame_valid_mask)
        out = {"frame_valid_mask": pred["frame_valid_mask"]}
        for k, v in pred.items():
            if k == "frame_valid_mask":
                continue
            out[k] = torch.sigmoid(v)
        return out

    def _masked_focal_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        alpha: float,
        gamma: float,
    ) -> torch.Tensor:
        mask = valid_mask > 0.5
        if mask.sum() == 0:
            return logits.new_tensor(0.0)

        bce = F.binary_cross_entropy_with_logits(logits[mask], target[mask], reduction="none")
        p = torch.sigmoid(logits[mask])
        pt = target[mask] * p + (1 - target[mask]) * (1 - p)
        alpha_t = target[mask] * alpha + (1 - target[mask]) * (1 - alpha)
        return (alpha_t * ((1 - pt) ** gamma) * bce).mean()

    def _masked_bce(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask > 0.5
        if logits.dim() == 3:
            mask = mask.unsqueeze(-1).expand_as(logits)

        if mask.sum() == 0:
            return logits.new_tensor(0.0)

        return F.binary_cross_entropy_with_logits(logits[mask], target[mask])

    def compute_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tg = batch["signal_targets"]
        alpha = float(self.cfg["stage2"]["focal_alpha"])
        gamma = float(self.cfg["stage2"]["focal_gamma"])
        valid = batch["frame_valid_mask"]

        losses = {}
        channel_map = {"u": 0, "a": 1}
        for prefix, ch in channel_map.items():
            for name in ["eot", "hold", "bot", "bc"]:
                losses[f"{prefix}_{name}"] = self._masked_focal_loss(
                    preds[f"{prefix}_{name}"],
                    tg[name][:, ch],
                    valid,
                    alpha,
                    gamma,
                )

            losses[f"{prefix}_vad"] = self._masked_bce(preds[f"{prefix}_vad"], tg["vad"][:, ch], valid)
            losses[f"{prefix}_fvad"] = self._masked_bce(preds[f"{prefix}_fvad"], tg["fvad"][:, ch], valid)

        loss = sum(losses.values()) / len(losses)
        out = {"loss": loss}
        out.update({k: v.detach() for k, v in losses.items()})
        return out