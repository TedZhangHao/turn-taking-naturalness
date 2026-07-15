from __future__ import annotations
from collections import Counter
import os
from pathlib import Path

import argparse
import json
from typing import Any

import joblib
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel

from dualturn.config import load_config
from dualturn.data.io import load_audio_and_meta
from dualturn.data.manifest import load_manifest
from dualturn.data.dataset_hf_official import normalize_official_repo_ids
from dualturn.data.vad import rms_vad
from dualturn.models.official_wrapper import DualTurnOfficialWrapper


FRAME_RATE = 12.5
CHUNK_FRAMES = 375
MS_PER_FRAME = 1000.0 / FRAME_RATE

SIGNAL_NAMES = [
    "eot_user", "eot_agent",
    "hold_user", "hold_agent",
    "bot_user", "bot_agent",
    "bc_user", "bc_agent",
    "vad_user", "vad_agent",
    "fvad_user_near", "fvad_user_far",
    "fvad_agent_near", "fvad_agent_far",
    "eot_minus_hold_user", "eot_minus_hold_agent",
]

CLASS_NAMES = [
    "start-talking",
    "continue-listening",
    "start-listening",
    "continue-talking",
    "backchannel",
]

FEAT_SUBSETS = {
    "FVAD": ["fvad_user_near", "fvad_user_far", "fvad_agent_near", "fvad_agent_far"],
    "Obvious": ["eot_user", "hold_user", "bot_user", "bc_user", "bc_agent"],
    "Core-8": ["eot_user", "eot_agent", "hold_user", "hold_agent", "bot_user", "bot_agent", "bc_user", "bc_agent"],
    "LR-all": SIGNAL_NAMES,
}


def apply_runtime_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    hf_cfg = cfg.setdefault("hf_official", {})
    paths_cfg = cfg.setdefault("paths", {})

    env_input_source = os.environ.get("HF_INPUT_SOURCE")
    if env_input_source:
        hf_cfg["input_source"] = env_input_source

    env_vad_source = os.environ.get("HF_VAD_SOURCE")
    if env_vad_source:
        hf_cfg["vad_source"] = env_vad_source

    env_output_dir = os.environ.get("HF_OUTPUT_DIR")
    if env_output_dir:
        paths_cfg["output_dir"] = env_output_dir

    return cfg


def get_model_impl(cfg: dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    return str(model_cfg.get("impl", model_cfg.get("stage2_impl", "legacy"))).lower()


def get_input_source(cfg: dict[str, Any]) -> str:
    source = str(cfg.get("hf_official", {}).get("input_source", "feature")).lower()
    if source not in {"feature", "audio"}:
        raise ValueError(f"Unsupported hf_official.input_source={source!r}; expected 'feature' or 'audio'.")
    return source


def get_vad_source(cfg: dict[str, Any]) -> str:
    source = str(cfg.get("hf_official", {}).get("vad_source", "official")).lower()
    if source not in {"official", "local"}:
        raise ValueError(f"Unsupported hf_official.vad_source={source!r}; expected 'official' or 'local'.")
    return source


def get_target_sample_rate(cfg: dict[str, Any]) -> int:
    return int(cfg.get("data", {}).get("target_sample_rate", 24000))


def get_samples_per_frame(cfg: dict[str, Any]) -> int:
    return int(cfg.get("data", {}).get("samples_per_frame", 1920))


def get_vad_threshold(cfg: dict[str, Any]) -> float:
    return float(cfg.get("data", {}).get("vad", {}).get("rms_threshold", 0.015))


def get_action_eval_repo_ids(cfg: dict[str, Any]) -> list[str]:
    repo_ids = cfg.get("data", {}).get("official_action_eval_repo_ids")
    if repo_ids:
        return normalize_official_repo_ids(repo_ids)

    repo_ids = cfg.get("data", {}).get("official_test_repo_ids")
    if repo_ids:
        return normalize_official_repo_ids(repo_ids)

    hf_repo = cfg.get("hf_official", {}).get("dataset_repo_id")
    if hf_repo:
        return [str(hf_repo)]

    return ["anyreach-ai/dualturn-otospeech-turn-taking"]


def _subset_indices(subset_name: str) -> list[int]:
    return [SIGNAL_NAMES.index(n) for n in FEAT_SUBSETS[subset_name]]


def _reshape_flat_2d(flat_arr, total_frames: int, dim: int, dtype) -> np.ndarray:
    arr = np.asarray(flat_arr, dtype=dtype)
    expected = total_frames * dim
    if arr.size != expected:
        raise RuntimeError(
            f"Expected flat array with {expected} values (= {total_frames}*{dim}), got {arr.size}."
        )
    return arr.reshape(total_frames, dim)


def _official_out_to_local_prob_dict(out, frame_valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "u_vad": out.vad_probs[..., 0],
        "a_vad": out.vad_probs[..., 1],
        "u_fvad": out.fvad_probs[..., :4],
        "a_fvad": out.fvad_probs[..., 4:8],
        "u_eot": out.eot_probs[..., 0],
        "a_eot": out.eot_probs[..., 1],
        "u_hold": out.hold_probs[..., 0],
        "a_hold": out.hold_probs[..., 1],
        "u_bot": out.bot_probs[..., 0],
        "a_bot": out.bot_probs[..., 1],
        "u_bc": out.bc_probs[..., 0],
        "a_bc": out.bc_probs[..., 1],
        "frame_valid_mask": frame_valid_mask,
    }


def build_local_manifest_index(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    candidate_paths: list[str] = []
    for key in ["all_manifest", "train_manifest", "val_manifest", "test_manifest"]:
        path = data_cfg.get(key)
        if path:
            candidate_paths.append(str(path))

    rows_by_session: dict[str, dict[str, Any]] = {}
    for path in candidate_paths:
        manifest_path = Path(path)
        if not manifest_path.exists():
            continue
        for row in load_manifest(manifest_path):
            session_id = str(row.get("session_id") or row.get("id") or "")
            if session_id and session_id not in rows_by_session:
                rows_by_session[session_id] = row
    return rows_by_session


def compute_local_vad(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target_sr = get_target_sample_rate(cfg)
    samples_per_frame = get_samples_per_frame(cfg)
    threshold = get_vad_threshold(cfg)

    audio, _, _ = load_audio_and_meta(row, target_sr)
    vad0 = rms_vad(audio[0], samples_per_frame, threshold).cpu().numpy().astype(np.float32)
    vad1 = rms_vad(audio[1], samples_per_frame, threshold).cpu().numpy().astype(np.float32)
    return vad0, vad1


def build_predictor(cfg: dict[str, Any], device: torch.device, stage2_ckpt: str | None):
    input_source = get_input_source(cfg)
    impl = get_model_impl(cfg)

    if input_source == "audio":
        if stage2_ckpt:
            print(
                "Warning: stage2_ckpt is ignored for input_source=audio; "
                "audio-mode event eval uses the public HF model directly."
            )
        repo_id = cfg.get("hf_official", {}).get(
            "model_repo_id",
            cfg.get("model", {}).get("official_repo_id", "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"),
        )
        precision_mode = str(cfg.get("precision", {}).get("mode", "fp32")).lower()
        if precision_mode in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif precision_mode in {"fp16", "float16", "half"}:
            dtype = torch.float16
        else:
            dtype = torch.float32

        model = DualTurnOfficialWrapper.from_pretrained(repo_id=repo_id, dtype=dtype).to(device).eval()
        print(f"Loaded official HF checkpoint for raw-audio inference: {repo_id}")
        return {
            "kind": "audio_hf",
            "model": model,
            "target_sr": get_target_sample_rate(cfg),
        }

    if impl == "official_source" and stage2_ckpt:
        from dualturn.models.stage2_official_source import Stage2OfficialSourceModel

        model = Stage2OfficialSourceModel(cfg).to(device)
        payload = torch.load(stage2_ckpt, map_location=device)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded official_source stage2 checkpoint: {stage2_ckpt}")
        if missing:
            print("Missing keys (first 10):", missing[:10])
        if unexpected:
            print("Unexpected keys (first 10):", unexpected[:10])
        model.eval()
        return {"kind": "official_source", "model": model}

    repo_id = cfg.get("hf_official", {}).get(
        "model_repo_id",
        cfg.get("model", {}).get("official_repo_id", "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"),
    )
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(device).eval()
    print(f"Loaded official HF checkpoint for feature inference: {repo_id}")
    return {"kind": "hf_feature", "model": model}


def _predict_chunk_probs(
    predictor: dict[str, Any],
    *,
    codes_ch0: torch.Tensor,
    codes_ch1: torch.Tensor,
    mimi_feat_ch0: torch.Tensor,
    mimi_feat_ch1: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    device: torch.device,
    audio: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    kind = predictor["kind"]
    model = predictor["model"]

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
            dtype=torch.bfloat16 if device.type == "cuda" else None,
        ):
            if kind == "official_source":
                return model.predict_signal_probs(
                    codes_ch0=codes_ch0,
                    codes_ch1=codes_ch1,
                    mimi_feat_ch0=mimi_feat_ch0,
                    mimi_feat_ch1=mimi_feat_ch1,
                    frame_valid_mask=frame_valid_mask,
                )

            if kind == "audio_hf":
                if audio is None:
                    raise ValueError("audio_hf predictor requires an audio tensor.")
                out = model(
                    audio=audio,
                    sr=int(predictor["target_sr"]),
                    return_hidden_states=False,
                )
                return _official_out_to_local_prob_dict(out, frame_valid_mask)

            out = model(
                audio=torch.zeros(codes_ch0.shape[0], 2, 1, device=device),
                mimi_feat_ch0=mimi_feat_ch0,
                mimi_feat_ch1=mimi_feat_ch1,
            )
            return _official_out_to_local_prob_dict(out, frame_valid_mask)


def run_session_inference(
    predictor: dict[str, Any],
    row: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    total_frames = int(row["num_frames"])
    chunk_frames = int(cfg.get("data", {}).get("window_frames", CHUNK_FRAMES))

    codes_ch0 = _reshape_flat_2d(row["codes_ch0"], total_frames, 8, np.int64)
    codes_ch1 = _reshape_flat_2d(row["codes_ch1"], total_frames, 8, np.int64)
    mimi_feat_ch0 = _reshape_flat_2d(row["mimi_feat_ch0"], total_frames, 512, np.float32)
    mimi_feat_ch1 = _reshape_flat_2d(row["mimi_feat_ch1"], total_frames, 512, np.float32)
    preds = {k: np.zeros((total_frames, 2), dtype=np.float32) for k in ["eot", "hold", "bot", "bc", "vad"]}
    preds["fvad"] = np.zeros((total_frames, 8), dtype=np.float32)

    for i in range(0, total_frames, chunk_frames):
        j = min(i + chunk_frames, total_frames)
        if j - i < 10:
            continue

        codes0 = torch.from_numpy(codes_ch0[i:j]).unsqueeze(0).to(device)
        codes1 = torch.from_numpy(codes_ch1[i:j]).unsqueeze(0).to(device)
        feat0 = torch.from_numpy(mimi_feat_ch0[i:j]).unsqueeze(0).to(device)
        feat1 = torch.from_numpy(mimi_feat_ch1[i:j]).unsqueeze(0).to(device)
        valid_mask = torch.ones(1, j - i, device=device, dtype=torch.float32)

        probs = _predict_chunk_probs(
            predictor,
            codes_ch0=codes0,
            codes_ch1=codes1,
            mimi_feat_ch0=feat0,
            mimi_feat_ch1=feat1,
            frame_valid_mask=valid_mask,
            device=device,
        )

        preds["eot"][i:j, 0] = probs["u_eot"].squeeze(0).detach().float().cpu().numpy()
        preds["eot"][i:j, 1] = probs["a_eot"].squeeze(0).detach().float().cpu().numpy()
        preds["hold"][i:j, 0] = probs["u_hold"].squeeze(0).detach().float().cpu().numpy()
        preds["hold"][i:j, 1] = probs["a_hold"].squeeze(0).detach().float().cpu().numpy()
        preds["bot"][i:j, 0] = probs["u_bot"].squeeze(0).detach().float().cpu().numpy()
        preds["bot"][i:j, 1] = probs["a_bot"].squeeze(0).detach().float().cpu().numpy()
        preds["bc"][i:j, 0] = probs["u_bc"].squeeze(0).detach().float().cpu().numpy()
        preds["bc"][i:j, 1] = probs["a_bc"].squeeze(0).detach().float().cpu().numpy()
        preds["vad"][i:j, 0] = probs["u_vad"].squeeze(0).detach().float().cpu().numpy()
        preds["vad"][i:j, 1] = probs["a_vad"].squeeze(0).detach().float().cpu().numpy()
        preds["fvad"][i:j, :4] = probs["u_fvad"].squeeze(0).detach().float().cpu().numpy()
        preds["fvad"][i:j, 4:8] = probs["a_fvad"].squeeze(0).detach().float().cpu().numpy()

    return preds


def run_audio_session_inference(
    predictor: dict[str, Any],
    row: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    target_sr = int(predictor["target_sr"])
    samples_per_frame = get_samples_per_frame(cfg)
    chunk_frames = int(cfg.get("data", {}).get("window_frames", CHUNK_FRAMES))
    chunk_samples = chunk_frames * samples_per_frame

    audio, _, _ = load_audio_and_meta(row, target_sr)
    total_frames = (audio.shape[-1] + samples_per_frame - 1) // samples_per_frame

    preds = {k: np.zeros((total_frames, 2), dtype=np.float32) for k in ["eot", "hold", "bot", "bc", "vad"]}
    preds["fvad"] = np.zeros((total_frames, 8), dtype=np.float32)

    for start_frame in range(0, total_frames, chunk_frames):
        start_sample = start_frame * samples_per_frame
        end_sample = min(audio.shape[-1], start_sample + chunk_samples)
        x_valid = audio[:, start_sample:end_sample]
        valid_num_samples = int(x_valid.shape[-1])
        if valid_num_samples <= 0:
            continue

        if valid_num_samples < chunk_samples:
            x = torch.nn.functional.pad(x_valid, (0, chunk_samples - valid_num_samples))
        else:
            x = x_valid[:, :chunk_samples]

        valid_frames = min(chunk_frames, (valid_num_samples + samples_per_frame - 1) // samples_per_frame)
        frame_valid_mask = torch.zeros(1, chunk_frames, dtype=torch.float32, device=device)
        frame_valid_mask[:, :valid_frames] = 1.0

        probs = _predict_chunk_probs(
            predictor,
            codes_ch0=torch.zeros(1, chunk_frames, 8, dtype=torch.long, device=device),
            codes_ch1=torch.zeros(1, chunk_frames, 8, dtype=torch.long, device=device),
            mimi_feat_ch0=torch.zeros(1, chunk_frames, 512, dtype=torch.float32, device=device),
            mimi_feat_ch1=torch.zeros(1, chunk_frames, 512, dtype=torch.float32, device=device),
            frame_valid_mask=frame_valid_mask,
            device=device,
            audio=x.unsqueeze(0).to(device),
        )

        pred_frames = int(probs["u_eot"].shape[1])
        fill_frames = min(valid_frames, pred_frames, total_frames - start_frame)
        stop = start_frame + fill_frames
        preds["eot"][start_frame:stop, 0] = probs["u_eot"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["eot"][start_frame:stop, 1] = probs["a_eot"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["hold"][start_frame:stop, 0] = probs["u_hold"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["hold"][start_frame:stop, 1] = probs["a_hold"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["bot"][start_frame:stop, 0] = probs["u_bot"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["bot"][start_frame:stop, 1] = probs["a_bot"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["bc"][start_frame:stop, 0] = probs["u_bc"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["bc"][start_frame:stop, 1] = probs["a_bc"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["vad"][start_frame:stop, 0] = probs["u_vad"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["vad"][start_frame:stop, 1] = probs["a_vad"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["fvad"][start_frame:stop, :4] = probs["u_fvad"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]
        preds["fvad"][start_frame:stop, 4:8] = probs["a_fvad"].squeeze(0).detach().float().cpu().numpy()[:fill_frames]

    return preds


def iter_hf_sessions(repo_ids: list[str], split: str, cache_dir: str | None = None):
    for repo_id in repo_ids:
        ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)
        for row in ds:
            yield repo_id, row


def extract_signals(preds, frame, user_ch, agent_ch, window=5):
    total_frames = preds["eot"].shape[0]
    lo = max(0, frame - window)
    hi = min(total_frames, frame + window + 1)

    def peak(key, ch):
        return float(preds[key][lo:hi, ch].max())

    def mean_val(key, ch):
        return float(preds[key][lo:hi, ch].mean())

    fvad = preds["fvad"][lo:hi]
    feats = {
        "eot_user": peak("eot", user_ch),
        "eot_agent": peak("eot", agent_ch),
        "hold_user": peak("hold", user_ch),
        "hold_agent": peak("hold", agent_ch),
        "bot_user": peak("bot", user_ch),
        "bot_agent": peak("bot", agent_ch),
        "bc_user": peak("bc", user_ch),
        "bc_agent": peak("bc", agent_ch),
        "vad_user": mean_val("vad", user_ch),
        "vad_agent": mean_val("vad", agent_ch),
        "fvad_user_near": float(fvad[:, user_ch * 4 : user_ch * 4 + 2].max()),
        "fvad_user_far": float(fvad[:, user_ch * 4 + 2 : user_ch * 4 + 4].max()),
        "fvad_agent_near": float(fvad[:, agent_ch * 4 : agent_ch * 4 + 2].max()),
        "fvad_agent_far": float(fvad[:, agent_ch * 4 + 2 : agent_ch * 4 + 4].max()),
        "eot_minus_hold_user": peak("eot", user_ch) - peak("hold", user_ch),
        "eot_minus_hold_agent": peak("eot", agent_ch) - peak("hold", agent_ch),
    }
    return [feats[n] for n in SIGNAL_NAMES]


def extract_signals_at_frame(preds, frame, user_ch, agent_ch):
    total_frames = preds["eot"].shape[0]
    if frame < 0 or frame >= total_frames:
        return [float("nan")] * len(SIGNAL_NAMES)
    feats = {
        "eot_user": float(preds["eot"][frame, user_ch]),
        "eot_agent": float(preds["eot"][frame, agent_ch]),
        "hold_user": float(preds["hold"][frame, user_ch]),
        "hold_agent": float(preds["hold"][frame, agent_ch]),
        "bot_user": float(preds["bot"][frame, user_ch]),
        "bot_agent": float(preds["bot"][frame, agent_ch]),
        "bc_user": float(preds["bc"][frame, user_ch]),
        "bc_agent": float(preds["bc"][frame, agent_ch]),
        "vad_user": float(preds["vad"][frame, user_ch]),
        "vad_agent": float(preds["vad"][frame, agent_ch]),
        "fvad_user_near": float(preds["fvad"][frame, user_ch * 4 : user_ch * 4 + 2].max()),
        "fvad_user_far": float(preds["fvad"][frame, user_ch * 4 + 2 : user_ch * 4 + 4].max()),
        "fvad_agent_near": float(preds["fvad"][frame, agent_ch * 4 : agent_ch * 4 + 2].max()),
        "fvad_agent_far": float(preds["fvad"][frame, agent_ch * 4 + 2 : agent_ch * 4 + 4].max()),
        "eot_minus_hold_user": float(preds["eot"][frame, user_ch]) - float(preds["hold"][frame, user_ch]),
        "eot_minus_hold_agent": float(preds["eot"][frame, agent_ch]) - float(preds["hold"][frame, agent_ch]),
    }
    return [feats[n] for n in SIGNAL_NAMES]


def find_speech_segments(vad, min_len=3):
    segments = []
    in_speech = False
    start = 0
    for t in range(len(vad)):
        if vad[t] > 0.5 and not in_speech:
            in_speech = True
            start = t
        elif vad[t] <= 0.5 and in_speech:
            in_speech = False
            if t - start >= min_len:
                segments.append((start, t))
    if in_speech and len(vad) - start >= min_len:
        segments.append((start, len(vad)))
    return segments


def extract_all_events(vad_user, vad_agent, preds, user_ch, agent_ch, total_frames):
    lookahead = 50
    hold_window = 25
    min_speech = 4
    bc_threshold = 12
    margin = 15

    features, labels = [], []

    user_segs = find_speech_segments(vad_user, min_len=min_speech)
    for seg_start, seg_end in user_segs:
        if seg_end >= total_frames - 5:
            continue
        is_shift = False
        is_hold = False
        for dt in range(1, lookahead + 1):
            ft = seg_end + dt
            if ft >= total_frames:
                break
            if vad_agent[ft] > 0.5:
                is_shift = True
                break
            if dt <= hold_window and vad_user[ft] > 0.5:
                is_hold = True
                break
        if is_shift:
            features.append(extract_signals(preds, seg_end, user_ch, agent_ch))
            labels.append(0)
        elif is_hold:
            features.append(extract_signals(preds, seg_end, user_ch, agent_ch))
            labels.append(1)

    for active_vad, passive_vad, active_ch, passive_ch in [
        (vad_user, vad_agent, user_ch, agent_ch),
        (vad_agent, vad_user, agent_ch, user_ch),
    ]:
        for t in range(1, total_frames):
            if passive_vad[t] > 0.5 and passive_vad[t - 1] <= 0.5 and active_vad[t] > 0.5:
                onset_len = 0
                for dt in range(t, min(t + 60, total_frames)):
                    if passive_vad[dt] > 0.5:
                        onset_len += 1
                    else:
                        break
                if onset_len < 2:
                    continue
                if onset_len >= bc_threshold:
                    features.append(extract_signals(preds, t, active_ch, passive_ch))
                    labels.append(2)
                else:
                    features.append(extract_signals(preds, t, active_ch, passive_ch))
                    labels.append(3)

    agent_segs = find_speech_segments(vad_agent, min_len=2)
    bc_frames = set()
    for seg_start, seg_end in agent_segs:
        if seg_end - seg_start >= bc_threshold:
            continue
        mid = (seg_start + seg_end) // 2
        if mid < 5 or mid >= total_frames - 5:
            continue
        if vad_user[max(0, seg_start - 3) : min(total_frames, seg_end + 3)].mean() <= 0.5:
            continue
        eval_frame = max(0, seg_start - 2)
        features.append(extract_signals(preds, eval_frame, user_ch, agent_ch))
        labels.append(4)
        for f in range(seg_start - margin, seg_end + margin):
            bc_frames.add(f)

    user_only = [
        t for t in range(50, total_frames - 50)
        if vad_user[t] > 0.5 and vad_agent[t] <= 0.5 and t not in bc_frames
    ]
    rng = np.random.RandomState(42)
    n_bc_pos = sum(1 for l in labels if l == 4)
    n_neg = min(n_bc_pos, len(user_only))
    if n_neg > 0 and user_only:
        for f in rng.choice(user_only, size=n_neg, replace=False):
            features.append(extract_signals(preds, int(f), user_ch, agent_ch))
            labels.append(1)

    return features, labels


def extract_anticipation_events(vad_user, vad_agent, preds, user_ch, agent_ch, total_frames):
    min_speech = 4
    lookahead = 50
    hold_window = 25
    pre_frames = 12
    post_frames = 6

    events = []
    user_segs = find_speech_segments(vad_user, min_len=min_speech)

    for seg_start, seg_end in user_segs:
        if seg_end >= total_frames - post_frames - 5 or seg_end < pre_frames + 5:
            continue

        is_shift = False
        is_hold = False
        for dt in range(1, lookahead + 1):
            ft = seg_end + dt
            if ft >= total_frames:
                break
            if vad_agent[ft] > 0.5:
                is_shift = True
                break
            if dt <= hold_window and vad_user[ft] > 0.5:
                is_hold = True
                break

        if not is_shift and not is_hold:
            continue

        label = 0 if is_shift else 1
        signals_at_offsets = {}
        for rel_f in range(-pre_frames, post_frames + 1):
            frame = seg_end + rel_f
            signals_at_offsets[rel_f] = extract_signals_at_frame(preds, frame, user_ch, agent_ch)
        events.append({"label": label, "signals_at_offsets": signals_at_offsets})

    return events


def process_split(
    predictor: dict[str, Any],
    cfg: dict[str, Any],
    repo_ids: list[str],
    split: str,
    device: torch.device,
    cache_dir: str | None = None,
    label: str = "",
    local_rows: dict[str, dict[str, Any]] | None = None,
    local_vad_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
):
    all_features, all_labels = [], []
    all_antic_events = []
    session_count = 0
    input_source = get_input_source(cfg)
    vad_source = get_vad_source(cfg)

    for repo_id, row in iter_hf_sessions(repo_ids, split, cache_dir=cache_dir):
        session_count += 1
        session_id = row.get("session_id", f"{repo_id}:{session_count}")
        local_row = None if local_rows is None else local_rows.get(str(session_id))

        if input_source == "audio":
            if local_row is None:
                print(f"  {label} skip {session_id}: no local audio row for input_source=audio")
                continue
            preds = run_audio_session_inference(predictor, local_row, cfg, device)
        else:
            preds = run_session_inference(predictor, row, cfg, device)

        if vad_source == "official":
            total_ref_frames = int(row["num_frames"])
            vad0 = np.asarray(row["vad_ch0"], dtype=np.float32)[:total_ref_frames]
            vad1 = np.asarray(row["vad_ch1"], dtype=np.float32)[:total_ref_frames]
        else:
            if local_row is None:
                print(f"  {label} skip {session_id}: no local audio row for vad_source=local")
                continue
            if local_vad_cache is None:
                local_vad_cache = {}
            cache_key = str(session_id)
            if cache_key not in local_vad_cache:
                local_vad_cache[cache_key] = compute_local_vad(local_row, cfg)
            vad0, vad1 = local_vad_cache[cache_key]

        total_frames = min(preds["eot"].shape[0], len(vad0), len(vad1))
        if total_frames < 500:
            print(f"  {label} skip {session_id}: only {total_frames} aligned frames")
            continue

        preds = {
            key: value[:total_frames]
            for key, value in preds.items()
        }
        vad0 = vad0[:total_frames]
        vad1 = vad1[:total_frames]

        for user_ch, agent_ch, vad_u, vad_a in [(0, 1, vad0, vad1), (1, 0, vad1, vad0)]:
            f, l = extract_all_events(vad_u, vad_a, preds, user_ch, agent_ch, total_frames)
            all_features.extend(f)
            all_labels.extend(l)
            antic = extract_anticipation_events(vad_u, vad_a, preds, user_ch, agent_ch, total_frames)
            all_antic_events.extend(antic)

        if session_count % 10 == 0:
            counts = Counter(all_labels)
            print(
                f"  {label} [{session_count:3d}] {session_id}  "
                f"ST={counts.get(0,0)} CL={counts.get(1,0)} SL={counts.get(2,0)} "
                f"CT={counts.get(3,0)} BC={counts.get(4,0)}  antic={len(all_antic_events)}"
            )

    X = np.array(all_features) if all_features else np.empty((0, len(SIGNAL_NAMES)))
    y = np.array(all_labels)
    return X, y, all_antic_events


def fit_5class(X_val, y_val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fitted = {}
    for subset_name, feat_names in FEAT_SUBSETS.items():
        idx = _subset_indices(subset_name)
        X_sub = X_val[:, idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        lr = LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=42,
            solver="lbfgs",
            class_weight="balanced",
        )
        lr.fit(X_scaled, y_val)

        fitted[subset_name] = {
            "scaler": scaler,
            "lr": lr,
            "feat_idx": idx,
            "feat_names": feat_names,
        }
    return fitted


def evaluate_5class(X_test, y_test, fitted):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

    results = {}
    for subset_name, fit in fitted.items():
        idx = fit["feat_idx"]
        X_sub = X_test[:, idx]
        X_scaled = fit["scaler"].transform(X_sub)

        y_pred = fit["lr"].predict(X_scaled)
        y_prob = fit["lr"].predict_proba(X_scaled)

        wf1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        bacc = balanced_accuracy_score(y_test, y_pred)

        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        per_class_f1_dict = {
            CLASS_NAMES[i]: float(per_class_f1[i])
            for i in range(len(CLASS_NAMES))
            if i < len(per_class_f1)
        }

        per_class_auc = {}
        for ci, cname in enumerate(CLASS_NAMES):
            if ci >= y_prob.shape[1]:
                continue
            binary_true = (y_test == ci).astype(int)
            if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
                per_class_auc[cname] = float("nan")
                continue
            per_class_auc[cname] = float(roc_auc_score(binary_true, y_prob[:, ci]))

        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
        coef_dict = {}
        for ci, cname in enumerate(CLASS_NAMES):
            if ci >= fit["lr"].coef_.shape[0]:
                continue
            coef_dict[cname] = {
                fit["feat_names"][fi]: float(fit["lr"].coef_[ci, fi])
                for fi in range(len(fit["feat_names"]))
            }

        results[subset_name] = {
            "weighted_f1": float(wf1),
            "macro_f1": float(macro_f1),
            "balanced_acc": float(bacc),
            "per_class_f1": per_class_f1_dict,
            "per_class_auc": per_class_auc,
            "confusion_matrix": cm.tolist(),
            "coefficients": coef_dict,
        }
    return results


def _find_best_f1_threshold(scores, labels, n_steps=200):
    from sklearn.metrics import f1_score

    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, n_steps):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _compute_f1(scores, labels, threshold):
    from sklearn.metrics import f1_score

    preds = (scores >= threshold).astype(int)
    return f1_score(labels, preds, zero_division=0)


def fit_anticipation(antic_events_val):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_t0 = np.array([ev["signals_at_offsets"][0] for ev in antic_events_val])
    y = np.array([ev["label"] for ev in antic_events_val])

    fitted = {}
    for subset_name in FEAT_SUBSETS:
        idx = _subset_indices(subset_name)
        X_sub = X_t0[:, idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(X_scaled, y)

        probs = lr.predict_proba(X_scaled)[:, 0]
        best_t = _find_best_f1_threshold(probs, (y == 0).astype(int))

        fitted[subset_name] = {
            "scaler": scaler,
            "lr": lr,
            "feat_idx": idx,
            "threshold": best_t,
        }
    return fitted


def evaluate_anticipation(antic_events_test, fitted_antic):
    from sklearn.metrics import roc_auc_score

    pre_frames = 12
    post_frames = 6
    offsets = list(range(-pre_frames, post_frames + 1))
    y_all = np.array([ev["label"] for ev in antic_events_test])
    y_binary = (y_all == 0).astype(int)

    results = {
        "n_events": int(len(antic_events_test)),
        "n_shift": int((y_all == 0).sum()),
        "n_hold": int((y_all == 1).sum()),
        "offsets_ms": [o * MS_PER_FRAME for o in offsets],
    }

    for subset_name, fit in fitted_antic.items():
        idx = fit["feat_idx"]
        f1_curve = []
        auc_curve = []

        for offset in offsets:
            X_at_t = np.array([ev["signals_at_offsets"][offset] for ev in antic_events_test])
            X_sub = X_at_t[:, idx]
            nan_mask = np.isnan(X_sub).any(axis=1)
            if nan_mask.sum() > len(nan_mask) * 0.5:
                f1_curve.append(float("nan"))
                auc_curve.append(float("nan"))
                continue

            X_clean = np.nan_to_num(X_sub, nan=0.0)
            X_scaled = fit["scaler"].transform(X_clean)
            probs = fit["lr"].predict_proba(X_scaled)[:, 0]

            f1_val = _compute_f1(probs, y_binary, fit["threshold"])
            try:
                auc_val = float(roc_auc_score(y_binary, probs))
            except ValueError:
                auc_val = float("nan")

            f1_curve.append(float(f1_val))
            auc_curve.append(auc_val)

        results[subset_name] = {"f1": f1_curve, "auc": auc_curve}

    return results


def print_results(results_5class, results_antic, y_test):
    counts = Counter(y_test)
    print("\n" + "=" * 80)
    print("  PART A: 5-Class Agent Action Classification (test set)")
    print("=" * 80)
    print(
        f"  Events: {len(y_test)}  "
        + "  ".join(f"{CLASS_NAMES[i]}={counts.get(i,0)}" for i in range(5))
    )

    print(f"\n  {'Metric':<25}", end="")
    for sn in FEAT_SUBSETS:
        n_feat = len(FEAT_SUBSETS[sn])
        print(f"  {sn+f' ({n_feat})':>15}", end="")
    print()
    print(f"  {'-'*25}" + f"  {'-'*15}" * len(FEAT_SUBSETS))

    for metric_name in ["weighted_f1", "macro_f1", "balanced_acc"]:
        print(f"  {metric_name:<25}", end="")
        for sn in FEAT_SUBSETS:
            val = results_5class[sn][metric_name]
            print(f"  {val:>15.4f}", end="")
        print()

    print(f"\n  Per-class F1:")
    print(f"  {'Class':<25}", end="")
    for sn in FEAT_SUBSETS:
        print(f"  {sn:>15}", end="")
    print()
    print(f"  {'-'*25}" + f"  {'-'*15}" * len(FEAT_SUBSETS))
    for cname in CLASS_NAMES:
        print(f"  {cname:<25}", end="")
        for sn in FEAT_SUBSETS:
            val = results_5class[sn]["per_class_f1"].get(cname, float("nan"))
            print(f"  {val:>15.4f}", end="")
        print()

    print(f"\n  Per-class AUC (one-vs-rest):")
    print(f"  {'Class':<25}", end="")
    for sn in FEAT_SUBSETS:
        print(f"  {sn:>15}", end="")
    print()
    print(f"  {'-'*25}" + f"  {'-'*15}" * len(FEAT_SUBSETS))
    for cname in CLASS_NAMES:
        print(f"  {cname:<25}", end="")
        for sn in FEAT_SUBSETS:
            val = results_5class[sn]["per_class_auc"].get(cname, float("nan"))
            if np.isnan(val):
                print(f"  {'N/A':>15}", end="")
            else:
                print(f"  {val:>15.4f}", end="")
        print()

    print(f"\n\n{'=' * 80}")
    print("  PART B: Anticipation -- Shift vs Hold F1 at each time offset (test set)")
    print("=" * 80)
    print(
        f"  Events: {results_antic['n_events']} "
        f"(shift={results_antic['n_shift']}, hold={results_antic['n_hold']})"
    )

    offsets_ms = results_antic["offsets_ms"]
    print(f"\n  {'Offset':>10}", end="")
    for sn in FEAT_SUBSETS:
        print(f"  {sn+' F1':>12}  {sn+' AUC':>12}", end="")
    print()
    print(f"  {'-'*10}" + f"  {'-'*12}  {'-'*12}" * len(FEAT_SUBSETS))

    for i, oms in enumerate(offsets_ms):
        if i % 2 != 0 and abs(oms) > 100:
            continue
        print(f"  {oms:>+8.0f}ms", end="")
        for sn in FEAT_SUBSETS:
            f1_val = results_antic[sn]["f1"][i]
            auc_val = results_antic[sn]["auc"][i]
            f1_str = f"{f1_val:.4f}" if not np.isnan(f1_val) else "N/A"
            auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "N/A"
            print(f"  {f1_str:>12}  {auc_str:>12}", end="")
        print()

    print(f"\n  Key operating points (LR-all):")
    lr_f1 = results_antic["LR-all"]["f1"]
    lr_auc = results_antic["LR-all"]["auc"]
    for target_ms in [-800, -560, -400, -240, 0]:
        target_f = int(target_ms / MS_PER_FRAME)
        if target_f in range(-12, 7):
            idx = target_f + 12
            if idx < len(lr_f1) and not np.isnan(lr_f1[idx]):
                print(f"    at {target_ms:>+5.0f}ms:  F1={lr_f1[idx]:.4f}  AUC={lr_auc[idx]:.4f}")


def print_table1_row(results_5class, results_antic, X_test, y_test, fitted_5class, label="Model"):
    from sklearn.metrics import f1_score

    lr_all = results_5class["LR-all"]
    fit = fitted_5class["LR-all"]
    X_sub = X_test[:, fit["feat_idx"]]
    X_scaled = fit["scaler"].transform(X_sub)
    y_pred = fit["lr"].predict(X_scaled)
    mask4 = y_test < 4
    wf1_4 = (
        f1_score(y_test[mask4], y_pred[mask4], average="weighted", zero_division=0)
        if mask4.sum() > 0
        else float("nan")
    )

    offsets_ms = results_antic["offsets_ms"]
    lr_all_auc = results_antic["LR-all"]["auc"]

    def auc_at(target_ms):
        for i, ms in enumerate(offsets_ms):
            if abs(ms - target_ms) < 1:
                return lr_all_auc[i]
        return float("nan")

    st = lr_all["per_class_auc"].get("start-talking", float("nan"))
    cl = lr_all["per_class_auc"].get("continue-listening", float("nan"))
    sl = lr_all["per_class_auc"].get("start-listening", float("nan"))
    ct = lr_all["per_class_auc"].get("continue-talking", float("nan"))
    bc = lr_all["per_class_auc"].get("backchannel", float("nan"))
    wf5 = lr_all["weighted_f1"]
    a240 = auc_at(-240.0)
    a400 = auc_at(-400.0)

    def fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "--"

    print(f"\n\n{'=' * 100}")
    print("  PAPER TABLE 1 ROW")
    print(f"{'=' * 100}")
    print("| Model | Feats | ST AUC | CL AUC | SL AUC | CT AUC | BC AUC | wF1-4 | wF1-5 | Ant@-240 | Ant@-400 |")
    print("|-------|-------|--------|--------|--------|--------|--------|-------|-------|----------|----------|")
    print(
        f"| {label} | 16 | {fmt(st)} | {fmt(cl)} | {fmt(sl)} | {fmt(ct)} | {fmt(bc)} | "
        f"{fmt(wf1_4)} | {fmt(wf5)} | {fmt(a240)} | {fmt(a400)} |"
    )
    print(f"{'=' * 100}")


def save_results(path: Path, results_5class, results_antic, y_test):
    payload = {
        "test_distribution": {int(k): int(v) for k, v in Counter(y_test).items()},
        "num_test_events": int(len(y_test)),
        "results_5class": results_5class,
        "results_anticipation": results_antic,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved official-style event metrics -> {path}")


def fit_probe_bundle(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    stage2_ckpt: str | None = None,
) -> dict[str, Any]:
    cfg = apply_runtime_overrides(cfg)
    cache_dir = cfg.get("hf_official", {}).get("cache_dir")
    repo_ids = get_action_eval_repo_ids(cfg)
    predictor = build_predictor(cfg, device, stage2_ckpt)
    impl = get_model_impl(cfg)
    input_source = get_input_source(cfg)
    vad_source = get_vad_source(cfg)
    local_rows = build_local_manifest_index(cfg) if input_source == "audio" or vad_source == "local" else {}
    local_vad_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("=" * 80)
    print("  Official-Style Probe Fitting -- 5-Class + Anticipation")
    print("=" * 80)
    print(f"  Model impl:    {impl}")
    print(f"  Input source:  {input_source}")
    print(f"  VAD source:    {vad_source}")
    print(f"  Dataset repos: {repo_ids}")

    print(f"\n{'-' * 80}")
    print("  STEP 1: Processing VAL sessions")
    print(f"{'-' * 80}")
    X_val, y_val, antic_val = process_split(
        predictor,
        cfg,
        repo_ids,
        split="val",
        device=device,
        cache_dir=cache_dir,
        label="VAL",
        local_rows=local_rows,
        local_vad_cache=local_vad_cache,
    )
    print(f"\n  Val totals: {len(y_val)} events, {len(antic_val)} anticipation events")
    print(f"  Class distribution: {dict(Counter(y_val))}")
    if len(y_val) == 0:
        raise RuntimeError("No validation action events were extracted.")
    if len(antic_val) == 0:
        raise RuntimeError("No validation anticipation events were extracted.")

    print(f"\n{'-' * 80}")
    print("  STEP 2: Fitting 5-class LRs on VAL")
    print(f"{'-' * 80}")
    fitted_5class = fit_5class(X_val, y_val)
    fitted_antic = fit_anticipation(antic_val)

    return {
        "format": "official_event_probe_v1",
        "model_impl": impl,
        "input_source": input_source,
        "vad_source": vad_source,
        "repo_ids": repo_ids,
        "fitted_5class": fitted_5class,
        "fitted_antic": fitted_antic,
        "val_num_events": int(len(y_val)),
        "val_distribution": {int(k): int(v) for k, v in Counter(y_val).items()},
        "val_num_anticipation_events": int(len(antic_val)),
    }


def save_probe_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Saved official-style probe -> {path}")


def load_probe_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format") != "official_event_probe_v1":
        raise ValueError(f"Probe at {path} is not an official_event_probe_v1 artifact.")
    return bundle


def evaluate_with_probe_bundle(
    cfg: dict[str, Any],
    bundle: dict[str, Any],
    *,
    device: torch.device,
    stage2_ckpt: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    cfg = apply_runtime_overrides(cfg)
    cache_dir = cfg.get("hf_official", {}).get("cache_dir")
    repo_ids = get_action_eval_repo_ids(cfg)
    predictor = build_predictor(cfg, device, stage2_ckpt)
    impl = get_model_impl(cfg)
    input_source = get_input_source(cfg)
    vad_source = get_vad_source(cfg)
    local_rows = build_local_manifest_index(cfg) if input_source == "audio" or vad_source == "local" else {}
    local_vad_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    bundle_input_source = bundle.get("input_source")
    bundle_vad_source = bundle.get("vad_source")
    if bundle_input_source is not None and bundle_input_source != input_source:
        print(f"Warning: probe bundle input_source={bundle_input_source} but current config uses {input_source}")
    if bundle_vad_source is not None and bundle_vad_source != vad_source:
        print(f"Warning: probe bundle vad_source={bundle_vad_source} but current config uses {vad_source}")

    print(f"\n{'-' * 80}")
    print("  STEP 3: Processing TEST sessions")
    print(f"{'-' * 80}")
    X_test, y_test, antic_test = process_split(
        predictor,
        cfg,
        repo_ids,
        split="test",
        device=device,
        cache_dir=cache_dir,
        label="TEST",
        local_rows=local_rows,
        local_vad_cache=local_vad_cache,
    )
    print(f"\n  Test totals: {len(y_test)} events, {len(antic_test)} anticipation events")
    print(f"  Class distribution: {dict(Counter(y_test))}")
    if len(y_test) == 0:
        raise RuntimeError("No test action events were extracted.")
    if len(antic_test) == 0:
        raise RuntimeError("No test anticipation events were extracted.")

    print(f"\n{'-' * 80}")
    print("  STEP 4: Evaluating on TEST")
    print(f"{'-' * 80}")
    fitted_5class = bundle["fitted_5class"]
    fitted_antic = bundle["fitted_antic"]
    results_5class = evaluate_5class(X_test, y_test, fitted_5class)
    results_antic = evaluate_anticipation(antic_test, fitted_antic)

    print_results(results_5class, results_antic, y_test)
    print_table1_row(results_5class, results_antic, X_test, y_test, fitted_5class, label=impl)

    out_path = output_path if output_path is not None else Path(cfg["paths"]["output_dir"]) / "artifacts" / "official_event_metrics.json"
    save_results(out_path, results_5class, results_antic, y_test)
    return {
        "X_test": X_test,
        "y_test": y_test,
        "results_5class": results_5class,
        "results_antic": results_antic,
        "output_path": out_path,
        "input_source": input_source,
        "vad_source": vad_source,
    }


def run_full_evaluation(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    stage2_ckpt: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    cfg = apply_runtime_overrides(cfg)
    bundle = fit_probe_bundle(cfg, device=device, stage2_ckpt=stage2_ckpt)
    return evaluate_with_probe_bundle(
        cfg,
        bundle,
        device=device,
        stage2_ckpt=stage2_ckpt,
        output_path=output_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage2-ckpt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_runtime_overrides(cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output) if args.output else None
    run_full_evaluation(cfg, device=device, stage2_ckpt=args.stage2_ckpt, output_path=output_path)


if __name__ == "__main__":
    main()
