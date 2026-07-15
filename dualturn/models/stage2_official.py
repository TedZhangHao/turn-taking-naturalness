from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from dualturn.models.official_wrapper import DualTurnOfficialWrapper


def _safe_logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.logit(x)


class Stage2OfficialModel(nn.Module):
    """
    Adapter model that keeps your CURRENT Stage-2 script interface, but
    internally uses the OFFICIAL public DualTurn source/model.

    External API kept compatible with your current scripts:
      - forward(audio, sample_mask, frame_valid_mask) -> dict of logits
      - predict_signal_probs(...)
      - compute_loss(preds, batch)

    Internal behavior:
      - calls the official public model via DualTurnOfficialWrapper
      - converts official probabilities back to logits so your existing
        BCEWithLogits / focal-loss training code can stay unchanged
      - maps official channel order to your current key names:
          ch0 -> u_*
          ch1 -> a_*
        and fvad:
          [:, :, :4]   -> u_fvad
          [:, :, 4:8]  -> a_fvad
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        mcfg = cfg["model"]
        precision_mode = str(cfg.get("precision", {}).get("mode", "fp32")).lower()
        if precision_mode in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif precision_mode in {"fp16", "float16", "half"}:
            dtype = torch.float16
        else:
            dtype = torch.float32

        repo_id = str(mcfg.get("official_repo_id", "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"))
        local_files_only = bool(mcfg.get("official_local_files_only", False))

        self.official = DualTurnOfficialWrapper.from_pretrained(
            repo_id=repo_id,
            dtype=dtype,
            local_files_only=local_files_only,
        )

        # Optional freezing policy:
        # default = freeze official model and only use it as fixed signal predictor.
        freeze_official = bool(mcfg.get("freeze_official", True))
        if freeze_official:
            for p in self.official.parameters():
                p.requires_grad = False
        else:
            for p in self.official.parameters():
                p.requires_grad = True

        self.target_sr = int(cfg["data"]["target_sample_rate"])

    def forward(
        self,
        audio: torch.Tensor,
        sample_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns LOGITS in the same key format as your current Stage2DualTurnModel:
          u_eot, a_eot, ..., u_vad, a_vad, u_fvad, a_fvad, frame_valid_mask

        NOTE:
        - Official source returns probabilities.
        - We convert them back to logits with logit(clamp(p)).
        """
        # Make sure padded tail stays silent when chunks are padded.
        if sample_mask is not None:
            audio = audio * sample_mask.unsqueeze(1).to(audio.dtype)

        out = self.official(
            audio=audio,
            sr=self.target_sr,
            return_hidden_states=False,
        )

        # Official output shapes:
        # vad_probs  : [B, T, 2]
        # fvad_probs : [B, T, 8]
        # eot/bot/hold/bc_probs : [B, T, 2]
        pred: dict[str, torch.Tensor] = {}

        pred["u_vad"] = _safe_logit(out.vad_probs[..., 0])
        pred["a_vad"] = _safe_logit(out.vad_probs[..., 1])

        pred["u_fvad"] = _safe_logit(out.fvad_probs[..., :4])
        pred["a_fvad"] = _safe_logit(out.fvad_probs[..., 4:8])

        pred["u_eot"] = _safe_logit(out.eot_probs[..., 0])
        pred["a_eot"] = _safe_logit(out.eot_probs[..., 1])

        pred["u_hold"] = _safe_logit(out.hold_probs[..., 0])
        pred["a_hold"] = _safe_logit(out.hold_probs[..., 1])

        pred["u_bot"] = _safe_logit(out.bot_probs[..., 0])
        pred["a_bot"] = _safe_logit(out.bot_probs[..., 1])

        pred["u_bc"] = _safe_logit(out.bc_probs[..., 0])
        pred["a_bc"] = _safe_logit(out.bc_probs[..., 1])

        T = pred["u_vad"].shape[1]
        pred["frame_valid_mask"] = frame_valid_mask[:, :T]
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
