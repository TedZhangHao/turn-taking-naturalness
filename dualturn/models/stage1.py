from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from dualturn.models.backbone import MimiQwenBackbone


class CodebookHead(nn.Module):
    def __init__(self, hidden_size: int, codebook_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, codebook_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1DualTurnModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = MimiQwenBackbone(cfg)

        hidden_size = int(self.backbone.qwen.config.hidden_size)
        self.num_quantizers = int(cfg["model"]["num_quantizers"])
        self.codebook_size = int(self.backbone.codebook_size)

        self.left_heads = nn.ModuleList(
            [CodebookHead(hidden_size, self.codebook_size) for _ in range(self.num_quantizers)]
        )
        self.right_heads = nn.ModuleList(
            [CodebookHead(hidden_size, self.codebook_size) for _ in range(self.num_quantizers)]
        )

    def _loss_channel(
        self,
        hidden: torch.Tensor,
        codes: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        heads: nn.ModuleList,
    ) -> torch.Tensor:
        hidden = hidden[:, :-1]
        targets = codes[:, :, 1:]
        valid = frame_valid_mask[:, 1:]

        T = min(hidden.shape[1], targets.shape[-1], valid.shape[1])
        hidden = hidden[:, :T]
        targets = targets[:, :, :T]
        valid = valid[:, :T]

        losses = []
        for q in range(self.num_quantizers):
            logits = heads[q](hidden)
            target = targets[:, q, :]

            flat_logits = logits.reshape(-1, self.codebook_size)
            flat_target = target.reshape(-1)
            flat_valid = valid.reshape(-1) > 0.5

            if flat_valid.sum() == 0:
                continue

            losses.append(F.cross_entropy(flat_logits[flat_valid], flat_target[flat_valid]))

        if not losses:
            return hidden.new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(
        self,
        audio: torch.Tensor,
        sample_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.backbone(audio=audio, sample_mask=sample_mask, frame_valid_mask=frame_valid_mask)
        left_loss = self._loss_channel(out.hidden, out.left_codes, frame_valid_mask, self.left_heads)
        right_loss = self._loss_channel(out.hidden, out.right_codes, frame_valid_mask, self.right_heads)
        loss = left_loss + right_loss

        return {
            "loss": loss,
            "left_code_loss": left_loss.detach(),
            "right_code_loss": right_loss.detach(),
        }