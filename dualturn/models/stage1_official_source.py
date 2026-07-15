from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dualturn.models.official_source_utils import (
    build_official_model,
    flatten_loss_output,
    make_codebook_weights,
)


class Stage1OfficialSourceModel(nn.Module):
    OFFICIAL_STAGE_CONFIG = "configs/stage1/pretrain_audio.yaml"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.official, stage_cfg = build_official_model(cfg, self.OFFICIAL_STAGE_CONFIG)
        self.official.freeze_for_stage("stage2_dual_task")

        training_cfg = stage_cfg.get("training", {})
        stage1_cfg = cfg.get("stage1", {})
        self.weight_codebook = float(stage1_cfg.get("weight_codebook", training_cfg.get("weight_codebook", 1.0)))
        self.weight_text = float(stage1_cfg.get("weight_text", training_cfg.get("weight_text", 0.0)))
        self.codebook_weights = stage1_cfg.get("codebook_weights", training_cfg.get("codebook_weights"))

    def set_full_finetune(self) -> None:
        self.official.freeze_for_stage("stage2_dual_task_ft")

    def forward(
        self,
        codes_ch0: torch.Tensor,
        codes_ch1: torch.Tensor,
        mimi_feat_ch0: torch.Tensor,
        mimi_feat_ch1: torch.Tensor,
    ) -> dict[str, Any]:
        out = self.official(
            codes_ch0=codes_ch0,
            codes_ch1=codes_ch1,
            mimi_feat_ch0=mimi_feat_ch0,
            mimi_feat_ch1=mimi_feat_ch1,
            mode="dual_task",
            weight_codebook=self.weight_codebook,
            weight_text=self.weight_text,
            codebook_weights=make_codebook_weights(self.codebook_weights, device=codes_ch0.device),
        )
        return flatten_loss_output(out)
