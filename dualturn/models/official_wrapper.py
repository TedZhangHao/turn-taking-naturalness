from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class DualTurnOfficialOutput:
    vad_probs: torch.Tensor
    fvad_probs: torch.Tensor
    eot_probs: torch.Tensor
    bot_probs: torch.Tensor
    hold_probs: torch.Tensor
    bc_probs: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    frame_mask: Optional[torch.Tensor] = None


class DualTurnOfficialWrapper(nn.Module):
    """
    Thin wrapper around the official Hugging Face implementation.

    This does NOT try to re-implement the public checkpoint architecture.
    It loads the vendor's own `modeling_dualturn.py` via `trust_remote_code=True`.
    Use this path when the goal is:
      - exact source alignment with the public repo
      - direct loading of the official `model.safetensors`
      - avoiding key-remap / shape-mismatch issues in custom re-implementations
    """

    def __init__(
        self,
        repo_id: str = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B",
        dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.model = AutoModel.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=local_files_only,
        )

    def forward(
        self,
        audio: torch.Tensor,
        sr: int = 24000,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> DualTurnOfficialOutput:
        """
        Supports:
        - single example: [2, T]
        - batched:        [B, 2, T]
        The public official model itself expects a single example [2, T],
        so we loop over batch here and stack outputs back.
        """

        def _one(example_audio: torch.Tensor):
            # example_audio: [2, T]
            out = self.model(example_audio, sr=sr, **kwargs)
            return out

        # ---- single example ----
        if audio.dim() == 2:
            out = _one(audio)
            return DualTurnOfficialOutput(
                vad_probs=out.vad_probs,
                fvad_probs=out.fvad_probs,
                eot_probs=out.eot_probs,
                bot_probs=out.bot_probs,
                hold_probs=out.hold_probs,
                bc_probs=out.bc_probs,
                hidden_states=getattr(out, "hidden_states", None) if return_hidden_states else None,
                frame_mask=getattr(out, "frame_mask", None),
            )

        # ---- batched ----
        if audio.dim() != 3 or audio.shape[1] != 2:
            raise ValueError(
                f"Expected audio shape [2, T] or [B, 2, T], got {tuple(audio.shape)}"
            )

        outs = []
        for i in range(audio.shape[0]):
            outs.append(_one(audio[i]))

        def _stack_attr(name: str):
            vals = [getattr(o, name) for o in outs]
            return torch.cat(vals, dim=0)

        hidden_states = None
        if return_hidden_states and getattr(outs[0], "hidden_states", None) is not None:
            hs = [getattr(o, "hidden_states") for o in outs]
            hidden_states = torch.cat(hs, dim=0)

        frame_mask = None
        if getattr(outs[0], "frame_mask", None) is not None:
            masks = [getattr(o, "frame_mask") for o in outs]
            frame_mask = torch.cat(masks, dim=0)

        return DualTurnOfficialOutput(
            vad_probs=_stack_attr("vad_probs"),
            fvad_probs=_stack_attr("fvad_probs"),
            eot_probs=_stack_attr("eot_probs"),
            bot_probs=_stack_attr("bot_probs"),
            hold_probs=_stack_attr("hold_probs"),
            bc_probs=_stack_attr("bc_probs"),
            hidden_states=hidden_states,
            frame_mask=frame_mask,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B",
        dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "DualTurnOfficialWrapper":
        return cls(
            repo_id=repo_id,
            dtype=dtype,
            device_map=device_map,
            local_files_only=local_files_only,
        )
