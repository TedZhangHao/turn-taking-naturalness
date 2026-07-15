from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, MimiModel


def _resolve_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


class ChannelMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BackboneOutput:
    hidden: torch.Tensor
    frame_mask: torch.Tensor
    left_codes: torch.Tensor
    right_codes: torch.Tensor


class MimiQwenBackbone(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        mcfg = cfg["model"]
        qwen_dtype = _resolve_dtype(mcfg.get("qwen_dtype", "bfloat16"))

        self.mimi = MimiModel.from_pretrained(mcfg["mimi_name"])
        self.mimi.eval()
        for p in self.mimi.parameters():
            p.requires_grad = False

        self.qwen = AutoModelForCausalLM.from_pretrained(
            mcfg["qwen_name"],
            torch_dtype=qwen_dtype,
        )

        if bool(mcfg.get("use_lora", True)):
            lora_cfg = LoraConfig(
                r=int(mcfg["lora_r"]),
                lora_alpha=int(mcfg["lora_alpha"]),
                lora_dropout=float(mcfg["lora_dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )
            self.qwen = get_peft_model(self.qwen, lora_cfg)

        self.num_quantizers = int(mcfg["num_quantizers"])
        self.cont_dim = int(mcfg["continuous_feature_dim"])
        self.codebook_size = int(getattr(self.mimi.config, "codebook_size", 2048))

        hidden_size = int(self.qwen.config.hidden_size)
        ch_proj_dim = int(mcfg["channel_proj_dim"])
        ch_hidden = int(mcfg["channel_mlp_hidden"])

        self.left_mlp = ChannelMLP(self.cont_dim, ch_hidden, ch_proj_dim)
        self.right_mlp = ChannelMLP(self.cont_dim, ch_hidden, ch_proj_dim)

        in_dim = ch_proj_dim * 2
        self.fusion = nn.Linear(in_dim, hidden_size) if in_dim != hidden_size else nn.Identity()

    def set_full_finetune(self) -> None:
        for p in self.qwen.parameters():
            p.requires_grad = True

    def _encode_codes(self, mono_audio: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        x = mono_audio.unsqueeze(1)
        padding_mask = sample_mask.unsqueeze(1).long()
        out = self.mimi.encode(
            x,
            padding_mask=padding_mask,
            return_dict=True,
            num_quantizers=self.num_quantizers,
        )
        return out.audio_codes[:, : self.num_quantizers]

    def _encode_continuous(self, mono_audio: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.mimi, "encoder"):
            raise RuntimeError(
                "This transformers build does not expose Mimi encoder internals. "
                "Use a recent transformers version where MimiModel has `.encoder`."
            )

        x = mono_audio.unsqueeze(1)
        padding_mask = sample_mask.unsqueeze(1).long()
        hs = self.mimi.encoder(x)

        if isinstance(hs, tuple):
            hs = hs[0]

        if hs.dim() != 3:
            raise RuntimeError(f"Unexpected Mimi encoder output shape: {tuple(hs.shape)}")

        if hs.shape[1] == self.cont_dim:
            hs = hs.transpose(1, 2)
        elif hs.shape[2] == self.cont_dim:
            pass
        else:
            raise RuntimeError(
                f"Unexpected Mimi encoder hidden size. Got {tuple(hs.shape)}, "
                f"expected one dimension to equal continuous_feature_dim={self.cont_dim}."
            )
        return hs

    def forward(
        self,
        audio: torch.Tensor,
        sample_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BackboneOutput:
        left_audio = audio[:, 0]
        right_audio = audio[:, 1]

        with torch.no_grad():
            left_cont = self._encode_continuous(left_audio, sample_mask)
            right_cont = self._encode_continuous(right_audio, sample_mask)
            left_codes = self._encode_codes(left_audio, sample_mask)
            right_codes = self._encode_codes(right_audio, sample_mask)

        T = min(left_cont.shape[1], right_cont.shape[1], left_codes.shape[-1], right_codes.shape[-1])
        left_cont = left_cont[:, :T]
        right_cont = right_cont[:, :T]
        left_codes = left_codes[:, :, :T]
        right_codes = right_codes[:, :, :T]

        left_proj = self.left_mlp(left_cont)
        right_proj = self.right_mlp(right_cont)
        fused = self.fusion(torch.cat([left_proj, right_proj], dim=-1))

        if frame_valid_mask is None:
            frame_mask = torch.ones((fused.shape[0], fused.shape[1]), dtype=torch.long, device=fused.device)
        else:
            frame_mask = frame_valid_mask[:, : fused.shape[1]].long().to(fused.device)

        outputs = self.qwen(
            inputs_embeds=fused.to(next(self.qwen.parameters()).dtype),
            attention_mask=frame_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1].float()

        return BackboneOutput(
            hidden=hidden,
            frame_mask=frame_mask,
            left_codes=left_codes,
            right_codes=right_codes,
        )