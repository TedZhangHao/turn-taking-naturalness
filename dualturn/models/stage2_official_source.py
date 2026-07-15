from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dualturn.models.official_source_utils import (
    build_official_model,
    flatten_loss_output,
    official_inference_to_local_probs,
    signal_targets_to_official_shapes,
)


class Stage2OfficialSourceModel(nn.Module):
    OFFICIAL_STAGE_CONFIG = "configs/stage2/model_A.yaml"

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.official, stage_cfg = build_official_model(cfg, self.OFFICIAL_STAGE_CONFIG)
        self.official.freeze_for_stage("stage3")

        training_cfg = stage_cfg.get("training", {})
        stage2_cfg = cfg.get("stage2", {})
        self.loss_kwargs = {
            "weight_eot": float(stage2_cfg.get("weight_eot", training_cfg.get("weight_eot", 1.0))),
            "weight_hold": float(stage2_cfg.get("weight_hold", training_cfg.get("weight_hold", 1.0))),
            "weight_bot": float(stage2_cfg.get("weight_bot", training_cfg.get("weight_bot", 1.0))),
            "weight_bc": float(stage2_cfg.get("weight_bc", training_cfg.get("weight_bc", 1.0))),
            "weight_vad": float(stage2_cfg.get("weight_vad", training_cfg.get("weight_vad", 1.0))),
            "weight_fvad": float(stage2_cfg.get("weight_fvad", training_cfg.get("weight_fvad", 0.0))),
            "weight_codebook": float(stage2_cfg.get("weight_codebook", training_cfg.get("weight_codebook", 0.0))),
            "sparse_loss_type": str(stage2_cfg.get("sparse_loss_type", training_cfg.get("sparse_loss_type", "focal"))),
            "sparse_pos_weight": float(stage2_cfg.get("sparse_pos_weight", training_cfg.get("sparse_pos_weight", 20.0))),
            "bc_pos_weight": float(stage2_cfg.get("bc_pos_weight", training_cfg.get("bc_pos_weight", 5.0))),
            "eot_pos_weight": float(stage2_cfg.get("eot_pos_weight", training_cfg.get("eot_pos_weight", 25.0))),
            "hold_pos_weight": float(stage2_cfg.get("hold_pos_weight", training_cfg.get("hold_pos_weight", 10.0))),
            "bot_pos_weight": float(stage2_cfg.get("bot_pos_weight", training_cfg.get("bot_pos_weight", 30.0))),
            "eot_alpha": float(stage2_cfg.get("eot_alpha", training_cfg.get("eot_alpha", 0.75))),
            "hold_alpha": float(stage2_cfg.get("hold_alpha", training_cfg.get("hold_alpha", 0.60))),
            "bot_alpha": float(stage2_cfg.get("bot_alpha", training_cfg.get("bot_alpha", 0.80))),
            "bc_alpha": float(stage2_cfg.get("bc_alpha", training_cfg.get("bc_alpha", 0.80))),
            "focal_gamma_sparse": float(stage2_cfg.get("focal_gamma_sparse", training_cfg.get("focal_gamma_sparse", 2.0))),
        }
        self.num_fvad_bins = int(self.official.num_fvad_bins)

    def configure_stage2_trainable(self) -> None:
        self.official.freeze_for_stage("stage3")

    def forward(
        self,
        codes_ch0: torch.Tensor,
        codes_ch1: torch.Tensor,
        mimi_feat_ch0: torch.Tensor,
        mimi_feat_ch1: torch.Tensor,
        signal_targets: dict[str, torch.Tensor],
        fvad_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        target_kwargs = signal_targets_to_official_shapes(signal_targets)
        out = self.official(
            codes_ch0=codes_ch0,
            codes_ch1=codes_ch1,
            mimi_feat_ch0=mimi_feat_ch0,
            mimi_feat_ch1=mimi_feat_ch1,
            mode="shift",
            fvad_mask=(fvad_mask > 0.5) if fvad_mask is not None else None,
            **target_kwargs,
            **self.loss_kwargs,
        )
        return flatten_loss_output(out)

    @torch.no_grad()
    def predict_signal_probs(
        self,
        codes_ch0: torch.Tensor,
        codes_ch1: torch.Tensor,
        mimi_feat_ch0: torch.Tensor,
        mimi_feat_ch1: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.official(
            codes_ch0=codes_ch0,
            codes_ch1=codes_ch1,
            mimi_feat_ch0=mimi_feat_ch0,
            mimi_feat_ch1=mimi_feat_ch1,
            mode="inference",
        )
        return official_inference_to_local_probs(
            out,
            frame_valid_mask=frame_valid_mask,
            num_fvad_bins=self.num_fvad_bins,
        )
