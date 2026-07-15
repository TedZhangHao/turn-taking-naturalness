from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch

from dualturn.models.official_source_loader import (
    get_official_repo_root,
    get_official_turntaking_model_class,
    load_official_yaml,
)


_MODEL_KEY_ALIASES = {
    "qwen_name": "qwen_model_name",
    "continuous_feature_dim": "mimi_feat_dim",
    "num_quantizers": "num_codebooks",
}


def _repo_root_str(cfg: dict[str, Any]) -> str:
    return str(get_official_repo_root(cfg))


def load_official_stage_config(cfg: dict[str, Any], rel_path: str) -> tuple[dict[str, Any], Path]:
    repo_root = get_official_repo_root(cfg)
    stage_cfg = load_official_yaml(str(repo_root), rel_path)
    return stage_cfg, repo_root


def build_official_model(cfg: dict[str, Any], rel_path: str):
    repo_root_str = _repo_root_str(cfg)
    TurnTakingModel = get_official_turntaking_model_class(repo_root_str)
    stage_cfg = load_official_yaml(repo_root_str, rel_path)

    official_model_cfg = dict(stage_cfg.get("model", {}))
    local_model_cfg = dict(cfg.get("model", {}))
    for src_key, dst_key in _MODEL_KEY_ALIASES.items():
        if src_key in local_model_cfg and dst_key not in local_model_cfg:
            local_model_cfg[dst_key] = local_model_cfg[src_key]

    allowed = {
        name
        for name in inspect.signature(TurnTakingModel.__init__).parameters
        if name != "self"
    }
    init_kwargs = {k: v for k, v in official_model_cfg.items() if k in allowed}
    for key, value in local_model_cfg.items():
        if key in allowed:
            init_kwargs[key] = value

    return TurnTakingModel(**init_kwargs), stage_cfg


def make_codebook_weights(
    weights: list[float] | tuple[float, ...] | torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if isinstance(weights, torch.Tensor):
        return weights.to(device=device, dtype=torch.float32)
    return torch.tensor(list(weights), device=device, dtype=torch.float32)


def flatten_loss_output(out: dict[str, Any]) -> dict[str, Any]:
    flat = {"loss": out["loss"]}
    for key, value in out.get("loss_dict", {}).items():
        flat[key] = value
    return flat


def signal_targets_to_official_shapes(signal_targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in ["eot", "hold", "bot", "bc", "vad"]:
        x = signal_targets[name]
        if x.dim() != 3:
            raise ValueError(f"Expected signal_targets[{name!r}] to have shape [B, 2, T], got {tuple(x.shape)}")
        out[f"{name}_targets"] = x.permute(0, 2, 1).contiguous()

    fvad = signal_targets["fvad"]
    if fvad.dim() != 4:
        raise ValueError(f"Expected signal_targets['fvad'] to have shape [B, 2, T, K], got {tuple(fvad.shape)}")
    bsz, num_ch, num_frames, num_bins = fvad.shape
    out["fvad_targets"] = fvad.permute(0, 2, 1, 3).reshape(bsz, num_frames, num_ch * num_bins).contiguous()
    return out


def official_inference_to_local_probs(
    out: dict[str, torch.Tensor],
    frame_valid_mask: torch.Tensor,
    *,
    num_fvad_bins: int,
) -> dict[str, torch.Tensor]:
    probs: dict[str, torch.Tensor] = {
        "u_eot": out["eot_probs"][..., 0],
        "a_eot": out["eot_probs"][..., 1],
        "u_hold": out["hold_probs"][..., 0],
        "a_hold": out["hold_probs"][..., 1],
        "u_bot": out["bot_probs"][..., 0],
        "a_bot": out["bot_probs"][..., 1],
        "u_bc": out["bc_probs"][..., 0],
        "a_bc": out["bc_probs"][..., 1],
        "u_vad": out["vad_probs"][..., 0],
        "a_vad": out["vad_probs"][..., 1],
        "u_fvad": out["fvad_probs"][..., :num_fvad_bins],
        "a_fvad": out["fvad_probs"][..., num_fvad_bins : 2 * num_fvad_bins],
    }
    probs["frame_valid_mask"] = frame_valid_mask[:, : probs["u_vad"].shape[1]]
    return probs
