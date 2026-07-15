from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, Qwen2ForCausalLM
from transformers.utils import ModelOutput


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DualTurnConfig(PretrainedConfig):
    model_type = "dualturn_qwen_mimi"

    def __init__(
        self,
        qwen_model_name: str   = "Qwen/Qwen2.5-0.5B",
        hidden_dim: int        = 896,
        mimi_feat_dim: int     = 512,
        mimi_sample_rate: int  = 24_000,
        mimi_frame_rate: float = 12.5,
        num_fvad_bins: int     = 4,
        head_type: str         = "linear",
        head_hidden_dim: int   = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qwen_model_name  = qwen_model_name
        self.hidden_dim       = hidden_dim
        self.mimi_feat_dim    = mimi_feat_dim
        self.mimi_sample_rate = mimi_sample_rate
        self.mimi_frame_rate  = mimi_frame_rate
        self.num_fvad_bins    = num_fvad_bins
        self.head_type        = head_type
        self.head_hidden_dim  = head_hidden_dim


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class DualTurnOutput(ModelOutput):
    """
    vad_probs   [B, T, 2]  P(speaking now)       — [:,0]=user  [:,1]=agent
    fvad_probs  [B, T, 8]  P(future speech)      — user 0:4  agent 4:8
                            horizons: 240 / 480 / 960 / 2000 ms
    eot_probs   [B, T, 2]  P(end of turn)         per channel
    bot_probs   [B, T, 2]  P(beginning of turn)   per channel
    hold_probs  [B, T, 2]  P(within-turn hold)    per channel
    bc_probs    [B, T, 2]  P(backchannel)         per channel
    """
    vad_probs:  torch.FloatTensor = None
    fvad_probs: torch.FloatTensor = None
    eot_probs:  torch.FloatTensor = None
    bot_probs:  torch.FloatTensor = None
    hold_probs: torch.FloatTensor = None
    bc_probs:   torch.FloatTensor = None


# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------

class _MimiProjection(nn.Module):
    def __init__(self, mimi_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(mimi_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, ch0: torch.Tensor, ch1: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([ch0, ch1], dim=-1))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualTurnModel(PreTrainedModel):
    """
    Dual-channel turn-taking model.
    Usage:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "anyreach-ai/dualturn-qwen2.5-mimi-0.5B",
            trust_remote_code=True,
        )
        model.eval()
        import torchaudio
        wav, sr = torchaudio.load("conversation.wav")  # [2, T], CH0=user CH1=agent
        with torch.no_grad():
            out = model(wav, sr=sr)
        # out.vad_probs   → [1, T, 2]
        # out.fvad_probs  → [1, T, 8]
        # out.eot_probs   → [1, T, 2]
        # out.bot_probs   → [1, T, 2]
    """

    config_class = DualTurnConfig

    def __init__(self, config: DualTurnConfig):
        super().__init__(config)
        D, N = config.hidden_dim, config.num_fvad_bins

        self.mimi_projection = _MimiProjection(config.mimi_feat_dim, D)

        self.backbone = Qwen2ForCausalLM.from_pretrained(
            config.qwen_model_name, torch_dtype=torch.float32)

        def _head():
            if config.head_type == "mlp":
                return nn.Sequential(
                    nn.Linear(D, config.head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(config.head_hidden_dim, 1),
                )
            return nn.Linear(D, 1)

        # VAD and FVAD are always Linear
        self.vad_head_ch0  = nn.Linear(D, 1)
        self.vad_head_ch1  = nn.Linear(D, 1)
        self.fvad_head     = nn.Linear(D, N * 2)
        # Sparse event heads follow head_type
        self.eot_head_ch0  = _head()
        self.eot_head_ch1  = _head()
        self.bot_head_ch0  = _head()
        self.bot_head_ch1  = _head()
        self.hold_head_ch0 = _head()
        self.hold_head_ch1 = _head()
        self.bc_head_ch0   = _head()
        self.bc_head_ch1   = _head()

        self._mimi_encoder = None

    # ------------------------------------------------------------------
    # from_pretrained — remaps research-repo key format
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        import json
        from pathlib import Path
        from huggingface_hub import hf_hub_download

        local = Path(pretrained_model_name_or_path)
        is_local = local.exists()

        def _get(filename):
            if is_local:
                return str(local / filename)
            return hf_hub_download(pretrained_model_name_or_path, filename)

        with open(_get("config.json")) as f:
            cfg_dict = json.load(f)

        valid = set(DualTurnConfig.__init__.__code__.co_varnames)
        config = DualTurnConfig(**{k: v for k, v in cfg_dict.items() if k in valid})

        model = cls(config)

        from safetensors.torch import load_file
        raw = load_file(_get("model.safetensors"), device="cpu")

        # Remap: research repo wraps Qwen2ForCausalLM inside QwenBackbone
        #   backbone.model.model.X   →  backbone.model.X
        #   backbone.model.lm_head.X →  backbone.lm_head.X
        remapped = {}
        for k, v in raw.items():
            if k.startswith("backbone.model.model."):
                k = "backbone.model." + k[len("backbone.model.model."):]
            elif k.startswith("backbone.model.lm_head."):
                k = "backbone.lm_head." + k[len("backbone.model.lm_head."):]
            remapped[k] = v

        missing, _ = model.load_state_dict(remapped, strict=False)
        real_missing = [k for k in missing
                        if not any(x in k for x in ("base_model", "audio_adapter",
                                                     "task_layer"))]
        if real_missing:
            print(f"[DualTurnModel] {len(real_missing)} missing keys: {real_missing[:3]}")

        device = kwargs.get("device_map", "cpu")
        if isinstance(device, str):
            model.to(device)
        return model.eval()

    # ------------------------------------------------------------------
    # Mimi encoder (lazy-loaded, not saved in weights)
    # ------------------------------------------------------------------

    def _get_mimi(self):
        if self._mimi_encoder is None:
            from transformers import MimiModel
            self._mimi_encoder = (
                MimiModel.from_pretrained("kyutai/mimi")
                .to(self.device).eval()
            )
        return self._mimi_encoder

    @torch.no_grad()
    def _encode_channel(self, wav_1d: torch.Tensor) -> torch.Tensor:
        mimi = self._get_mimi()
        x    = wav_1d.unsqueeze(0).unsqueeze(0).to(self.device)
        enc  = mimi.encoder(x)
        et   = mimi.encoder_transformer(enc.transpose(1, 2))
        if hasattr(et, "last_hidden_state"):
            et = et.last_hidden_state
        return mimi.downsample(et.transpose(1, 2)).squeeze(0).T.float()  # [T, 512]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        audio: torch.Tensor,
        sr: int = 24_000,
        mimi_feat_ch0: Optional[torch.Tensor] = None,
        mimi_feat_ch1: Optional[torch.Tensor] = None,
    ) -> DualTurnOutput:
        """
        Args:
            audio: [2, T_samples] stereo tensor, CH0=user CH1=agent.
                   Any sample rate — resampled internally to 24 kHz.
            sr:    Sample rate of `audio`.
            mimi_feat_ch0/ch1: [B, T, 512] pre-extracted Mimi features.
                   Skips audio encoding when provided.
        """
        if mimi_feat_ch0 is None:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).repeat(2, 1)
            if sr != self.config.mimi_sample_rate:
                import torchaudio
                audio = torchaudio.functional.resample(
                    audio, sr, self.config.mimi_sample_rate)
            feat0 = self._encode_channel(audio[0])
            feat1 = self._encode_channel(audio[1])
            T     = min(feat0.shape[0], feat1.shape[0])
            mimi_feat_ch0 = feat0[:T].unsqueeze(0)
            mimi_feat_ch1 = feat1[:T].unsqueeze(0)

        embeddings = self.mimi_projection(mimi_feat_ch0, mimi_feat_ch1)

        out = self.backbone(
            inputs_embeds=embeddings,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[-1]

        def _two(h0, h1):
            return torch.sigmoid(
                torch.stack([h0(h).squeeze(-1), h1(h).squeeze(-1)], dim=-1))

        return DualTurnOutput(
            vad_probs  = _two(self.vad_head_ch0,  self.vad_head_ch1),
            fvad_probs = torch.sigmoid(self.fvad_head(h)),
            eot_probs  = _two(self.eot_head_ch0,  self.eot_head_ch1),
            bot_probs  = _two(self.bot_head_ch0,  self.bot_head_ch1),
            hold_probs = _two(self.hold_head_ch0, self.hold_head_ch1),
            bc_probs   = _two(self.bc_head_ch0,   self.bc_head_ch1),
        )
