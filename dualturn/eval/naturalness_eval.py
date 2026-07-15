from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from dualturn.data.io import load_audio_and_meta
from dualturn.data.labels import NO_ACTION, derive_action_targets
from dualturn.data.manifest import load_manifest
from dualturn.data.vad import rms_vad
from dualturn.eval.action_naturalness_eval import action_proba_from_probe
from dualturn.eval.official_event_eval import apply_runtime_overrides, build_predictor, run_audio_session_inference


ACTION_NAMES = ["ST", "CL", "SL", "CT", "BC"]
ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}


def _frame_ms(cfg: dict[str, Any]) -> float:
    samples_per_frame = int(cfg.get("data", {}).get("samples_per_frame", 1920))
    sr = int(cfg.get("data", {}).get("target_sample_rate", 24000))
    return 1000.0 * samples_per_frame / max(sr, 1)


def _frame_hz(cfg: dict[str, Any]) -> float:
    return 1000.0 / max(_frame_ms(cfg), 1e-8)


def _threshold(cfg: dict[str, Any]) -> float:
    return float(cfg.get("data", {}).get("vad", {}).get("rms_threshold", 0.015))


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(xs: list[float], default: float = 0.0) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float(default)


def _segments(vad: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    active = False
    start = 0
    for i, v in enumerate(vad):
        if v > 0.5 and not active:
            start = i
            active = True
        elif v <= 0.5 and active:
            if i - start >= min_len:
                segs.append((start, i))
            active = False
    if active and len(vad) - start >= min_len:
        segs.append((start, len(vad)))
    return segs


def _binary_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    active = False
    start = 0
    for i, v in enumerate(mask.astype(bool)):
        if v and not active:
            start = i
            active = True
        elif not v and active:
            runs.append((start, i))
            active = False
    if active:
        runs.append((start, len(mask)))
    return runs


def _slice_max(x: np.ndarray, lo: int, hi: int) -> float:
    lo = max(0, int(lo))
    hi = min(len(x), int(hi))
    if hi <= lo:
        return 0.0
    return float(np.max(x[lo:hi]))


def _slice_mean(x: np.ndarray, lo: int, hi: int) -> float:
    lo = max(0, int(lo))
    hi = min(len(x), int(hi))
    if hi <= lo:
        return 0.0
    return float(np.mean(x[lo:hi]))


def _score_100_to_5(x: float) -> float:
    x = float(np.clip(x, 0.0, 100.0))
    return 1.0 + 4.0 * (x / 100.0)


def _exp_target_score(value: float, target: float, scale: float) -> float:
    return float(100.0 * math.exp(-abs(float(value) - float(target)) / max(float(scale), 1e-8)))


def _upper_bound_score(value: float, good_until: float, scale: float) -> float:
    value = float(value)
    if value <= good_until:
        return 100.0
    return float(100.0 * math.exp(-(value - good_until) / max(float(scale), 1e-8)))


def _mean_prob(x: np.ndarray, action_names: list[str], lo: int, hi: int) -> float:
    lo = max(0, int(lo))
    hi = min(x.shape[0], int(hi))
    if hi <= lo:
        return 0.0
    cols = [ACTION_TO_ID[name] for name in action_names]
    return float(x[lo:hi, cols].max(axis=1).mean())


def _preds_to_local_probs(preds: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    t = int(preds["eot"].shape[0])
    return {
        "u_eot": torch.from_numpy(preds["eot"][:, 0][None, :].astype(np.float32)),
        "a_eot": torch.from_numpy(preds["eot"][:, 1][None, :].astype(np.float32)),
        "u_hold": torch.from_numpy(preds["hold"][:, 0][None, :].astype(np.float32)),
        "a_hold": torch.from_numpy(preds["hold"][:, 1][None, :].astype(np.float32)),
        "u_bot": torch.from_numpy(preds["bot"][:, 0][None, :].astype(np.float32)),
        "a_bot": torch.from_numpy(preds["bot"][:, 1][None, :].astype(np.float32)),
        "u_bc": torch.from_numpy(preds["bc"][:, 0][None, :].astype(np.float32)),
        "a_bc": torch.from_numpy(preds["bc"][:, 1][None, :].astype(np.float32)),
        "u_vad": torch.from_numpy(preds["vad"][:, 0][None, :].astype(np.float32)),
        "a_vad": torch.from_numpy(preds["vad"][:, 1][None, :].astype(np.float32)),
        "u_fvad": torch.from_numpy(preds["fvad"][:, :4][None, :, :].astype(np.float32)),
        "a_fvad": torch.from_numpy(preds["fvad"][:, 4:8][None, :, :].astype(np.float32)),
        "frame_valid_mask": torch.ones(1, t, dtype=torch.float32),
    }


def _get_probe_path(cfg: dict[str, Any]) -> str | None:
    ncfg = cfg.get("naturalness", {})
    return (
        ncfg.get("action_probe_path")
        or cfg.get("probe", {}).get("path")
        or cfg.get("action_naturalness", {}).get("probe_path")
    )


def compute_local_vad_from_row(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target_sr = int(cfg.get("data", {}).get("target_sample_rate", 24000))
    samples_per_frame = int(cfg.get("data", {}).get("samples_per_frame", 1920))
    thr = _threshold(cfg)
    audio, _, _ = load_audio_and_meta(row, target_sr)
    vad0 = rms_vad(audio[0], samples_per_frame, thr).cpu().numpy().astype(np.float32)
    vad1 = rms_vad(audio[1], samples_per_frame, thr).cpu().numpy().astype(np.float32)
    return vad0, vad1


def _find_switch_events(
    own: list[tuple[int, int]],
    other: list[tuple[int, int]],
    from_ch: int,
    to_ch: int,
    max_gap_frames: int,
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    for s, e in own:
        for os, oe in other:
            if os <= s:
                continue
            if os > e + max_gap_frames:
                break
            events.append(
                {
                    "from_ch": from_ch,
                    "to_ch": to_ch,
                    "from_start": s,
                    "from_end": e,
                    "to_start": os,
                    "to_end": oe,
                    "gap_frames": os - e,
                }
            )
            break
    return events


def _find_hold_events(
    own: list[tuple[int, int]],
    other_vad: np.ndarray,
    hold_ch: int,
    max_gap_frames: int,
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    for idx in range(len(own) - 1):
        s, e = own[idx]
        ns, ne = own[idx + 1]
        gap = ns - e
        if gap < 0 or gap > max_gap_frames:
            continue
        if _slice_mean(other_vad, e, ns) > 0.1:
            continue
        events.append(
            {
                "hold_ch": hold_ch,
                "next_same_start": ns,
                "curr_start": s,
                "curr_end": e,
                "gap_frames": gap,
            }
        )
    return events


def _find_backchannel_events(
    speaker_vad: np.ndarray,
    listener_vad: np.ndarray,
    listener_ch: int,
    max_bc_frames: int,
) -> list[dict[str, int]]:
    events: list[dict[str, int]] = []
    for s, e in _segments(listener_vad, min_len=1):
        dur = e - s
        if dur <= 0 or dur > max_bc_frames:
            continue
        if _slice_mean(speaker_vad, s, e) < 0.5:
            continue
        events.append({"listener_ch": listener_ch, "start": s, "end": e, "dur_frames": dur})
    return events


def _is_bc_like_overlap(
    run: tuple[int, int],
    preds: dict[str, np.ndarray],
    cfg: dict[str, Any],
    frame_ms: float,
) -> bool:
    scfg = cfg.get("naturalness", {})
    max_bc_ms = float(scfg.get("bc_like_max_duration_ms", scfg.get("bc_max_duration_ms", 700.0)))
    bc_thr = float(scfg.get("bc_pred_threshold", 0.35))
    s, e = run
    dur_ms = (e - s) * frame_ms
    if dur_ms > max_bc_ms:
        return False
    return max(_slice_max(preds["bc"][:, 0], s - 1, e + 1), _slice_max(preds["bc"][:, 1], s - 1, e + 1)) >= bc_thr


def _action_stability_score(
    action_probs: np.ndarray,
    valid_mask: np.ndarray,
    include_bc: bool,
    cfg: dict[str, Any],
) -> float:
    n = min(len(valid_mask), action_probs.shape[0])
    if n == 0:
        return 50.0
    p = action_probs[:n].astype(np.float64).copy()
    mask = valid_mask[:n].astype(bool)
    if not include_bc:
        p[:, ACTION_TO_ID["BC"]] = 0.0
    denom = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    p = p / denom
    p = p[mask]
    if p.size == 0:
        return 50.0

    classes = p.shape[1]
    conf = p.max(axis=1)
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1) / max(math.log(classes), 1e-8)
    labels = p.argmax(axis=1)
    flip_rate = float(np.mean(labels[1:] != labels[:-1])) if len(labels) >= 2 else 0.0

    scfg = cfg.get("naturalness", {})
    conf_score = 100.0 * float(conf.mean())
    entropy_score = 100.0 * float(np.clip(1.0 - entropy.mean(), 0.0, 1.0))
    flip_score = _upper_bound_score(
        flip_rate,
        float(scfg.get("stability_flip_good", 0.12)),
        float(scfg.get("stability_flip_scale", 0.20)),
    )
    return _safe_mean([conf_score, entropy_score, flip_score], default=50.0)


def _switch_score(
    switch_events: list[dict[str, int]],
    preds: dict[str, np.ndarray],
    action_probs: np.ndarray,
    frame_ms: float,
    cfg: dict[str, Any],
) -> float:
    scfg = cfg.get("naturalness", {})
    pref_gap_ms = float(scfg.get("preferred_gap_ms", 240.0))
    gap_scale_ms = float(scfg.get("switch_gap_scale_ms", 360.0))
    overlap_scale_ms = float(scfg.get("overlap_scale_ms", 500.0))
    short_overlap_ok_ms = float(scfg.get("short_overlap_ok_ms", 120.0))
    vals = []
    for ev in switch_events:
        boundary = max(0, ev["from_end"] - 1)
        onset = ev["to_start"]
        gap_ms = ev["gap_frames"] * frame_ms
        from_ch = ev["from_ch"]
        to_ch = ev["to_ch"]
        eot = _slice_max(preds["eot"][:, from_ch], boundary - 2, boundary + 3)
        hold = _slice_max(preds["hold"][:, from_ch], boundary - 2, boundary + 3)
        bot = _slice_max(preds["bot"][:, to_ch], onset - 2, onset + 4)
        shift_action = _mean_prob(action_probs, ["ST", "SL"], onset - 2, onset + 3)
        boundary_score = 100.0 * np.clip((eot + (1.0 - hold) + bot + shift_action) / 4.0, 0.0, 1.0)
        if gap_ms >= 0:
            gap_score = _exp_target_score(gap_ms, pref_gap_ms, gap_scale_ms)
        else:
            gap_score = _upper_bound_score(abs(gap_ms), short_overlap_ok_ms, overlap_scale_ms)
        vals.append(0.65 * boundary_score + 0.35 * gap_score)
    return _safe_mean(vals, default=60.0)


def _hold_score(
    hold_events: list[dict[str, int]],
    preds: dict[str, np.ndarray],
    action_probs: np.ndarray,
    frame_ms: float,
    cfg: dict[str, Any],
) -> float:
    scfg = cfg.get("naturalness", {})
    hold_gap_ok_ms = float(scfg.get("hold_gap_ok_ms", 450.0))
    hold_gap_scale_ms = float(scfg.get("hold_gap_scale_ms", 700.0))
    vals = []
    for ev in hold_events:
        boundary = max(0, ev["curr_end"] - 1)
        hold_ch = ev["hold_ch"]
        other_ch = 1 - hold_ch
        gap_ms = ev["gap_frames"] * frame_ms
        hold = _slice_max(preds["hold"][:, hold_ch], boundary - 2, boundary + 3)
        eot = _slice_max(preds["eot"][:, hold_ch], boundary - 2, boundary + 3)
        other_bot = _slice_max(preds["bot"][:, other_ch], boundary - 1, boundary + 5)
        continue_action = _mean_prob(action_probs, ["CT", "CL"], boundary - 1, boundary + 4)
        gap_score = _upper_bound_score(gap_ms, hold_gap_ok_ms, hold_gap_scale_ms)
        vals.append(
            100.0
            * np.clip((hold + (1.0 - eot) + (1.0 - other_bot) + continue_action) / 4.0, 0.0, 1.0)
            * 0.7
            + 0.3 * gap_score
        )
    return _safe_mean(vals, default=65.0)


def _rhythm_score(
    switch_events: list[dict[str, int]],
    silence_runs: list[tuple[int, int]],
    frame_ms: float,
    cfg: dict[str, Any],
) -> float:
    scfg = cfg.get("naturalness", {})
    pref_gap_ms = float(scfg.get("preferred_gap_ms", 240.0))
    gap_scale_ms = float(scfg.get("switch_gap_scale_ms", 360.0))
    long_silence_ms = float(scfg.get("long_silence_ms", 1200.0))
    silence_scale_ms = float(scfg.get("silence_scale_ms", 1000.0))

    pos_gaps = [ev["gap_frames"] * frame_ms for ev in switch_events if ev["gap_frames"] >= 0]
    gap_score = _safe_mean([_exp_target_score(g, pref_gap_ms, gap_scale_ms) for g in pos_gaps], default=75.0)

    silence_durs = [(e - s) * frame_ms for s, e in silence_runs]
    silence_score = _safe_mean(
        [_upper_bound_score(d, long_silence_ms, silence_scale_ms) for d in silence_durs],
        default=100.0,
    )
    return 0.6 * gap_score + 0.4 * silence_score


def _overlap_score(
    overlap_runs: list[tuple[int, int]],
    preds: dict[str, np.ndarray],
    cfg: dict[str, Any],
    frame_ms: float,
    include_bc: bool,
    total_frames: int,
) -> tuple[float, dict[str, float]]:
    scfg = cfg.get("naturalness", {})
    short_overlap_ok_ms = float(scfg.get("short_overlap_ok_ms", 120.0))
    long_overlap_ms = float(scfg.get("long_overlap_ms", 700.0))
    overlap_scale_ms = float(scfg.get("overlap_scale_ms", 500.0))
    overlap_ratio_good = float(scfg.get("overlap_ratio_good", 0.04))
    overlap_ratio_scale = float(scfg.get("overlap_ratio_scale", 0.08))

    penalty_durs = []
    bc_like_count = 0
    for run in overlap_runs:
        dur_ms = (run[1] - run[0]) * frame_ms
        if _is_bc_like_overlap(run, preds, cfg, frame_ms):
            bc_like_count += 1
            if include_bc:
                penalty_durs.append(0.35 * dur_ms)
            continue
        penalty_durs.append(dur_ms)

    penalty_ratio = float(sum(penalty_durs) / max(total_frames * frame_ms, 1e-8))
    mean_penalty_ms = _safe_mean(penalty_durs, default=0.0)
    max_penalty_ms = max([0.0] + penalty_durs)
    long_penalty_count = sum(1 for d in penalty_durs if d > long_overlap_ms)

    score = _safe_mean(
        [
            _upper_bound_score(mean_penalty_ms, short_overlap_ok_ms, overlap_scale_ms),
            _upper_bound_score(max_penalty_ms, long_overlap_ms, overlap_scale_ms),
            _upper_bound_score(penalty_ratio, overlap_ratio_good, overlap_ratio_scale),
        ],
        default=80.0,
    )
    stats = {
        "bc_like_overlap_count": float(bc_like_count),
        "penalty_overlap_mean_ms": float(mean_penalty_ms),
        "penalty_overlap_max_ms": float(max_penalty_ms),
        "penalty_overlap_ratio": float(penalty_ratio),
        "long_penalty_overlap_count": float(long_penalty_count),
    }
    return score, stats


def _backchannel_score(
    bc_events: list[dict[str, int]],
    preds: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    frame_ms: float,
    cfg: dict[str, Any],
) -> float:
    scfg = cfg.get("naturalness", {})
    bc_max_ms = float(scfg.get("bc_max_duration_ms", 700.0))
    bc_scale_ms = float(scfg.get("bc_duration_scale_ms", 500.0))
    vals = []
    for ev in bc_events:
        dur_ms = ev["dur_frames"] * frame_ms
        ch = ev["listener_ch"]
        bc_prob = _slice_max(preds["bc"][:, ch], ev["start"] - 1, ev["end"] + 1)
        dur_score = _upper_bound_score(dur_ms, bc_max_ms, bc_scale_ms)
        vals.append(0.5 * (100.0 * bc_prob) + 0.5 * dur_score)

    if len(valid_mask) > 0:
        idle_bc = []
        for ch in [0, 1]:
            idle_bc.append(float(preds["bc"][:, ch][valid_mask].mean()) if valid_mask.any() else 0.0)
        false_pressure_score = _upper_bound_score(max(idle_bc), 0.25, 0.20)
    else:
        false_pressure_score = 80.0

    if not vals:
        return float(0.7 * 75.0 + 0.3 * false_pressure_score)
    return float(0.75 * _safe_mean(vals, default=75.0) + 0.25 * false_pressure_score)


def _derive_support_targets(vad0: np.ndarray, vad1: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    t = min(len(vad0), len(vad1))
    signals = {
        "vad": torch.from_numpy(np.stack([vad0[:t], vad1[:t]], axis=0).astype(np.float32)),
        "frame_valid_mask": torch.ones(t, dtype=torch.float32),
    }
    return derive_action_targets(signals, frame_hz=_frame_hz(cfg)).detach().cpu().numpy().astype(np.int64)


def _distribution_score(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    if len(targets) == 0:
        return 50.0
    pred_dist = probs.mean(axis=0)
    pred_dist = pred_dist / max(float(pred_dist.sum()), 1e-12)
    target_dist = np.bincount(targets, minlength=num_classes).astype(np.float64)
    target_dist = target_dist / max(float(target_dist.sum()), 1e-12)
    l1 = float(np.abs(pred_dist - target_dist).sum())
    return float(100.0 * max(0.0, 1.0 - 0.5 * l1))


def _support_frame_action_score(
    action_probs: np.ndarray,
    targets: np.ndarray,
    include_bc: bool,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    mask = targets != NO_ACTION
    if not include_bc:
        mask &= targets != ACTION_TO_ID["BC"]

    if mask.sum() == 0:
        return {
            "raw_score": 50.0,
            "support_frames": 0,
            "components": {
                "alignment": 50.0,
                "margin": 50.0,
                "accuracy": 50.0,
                "confidence": 50.0,
                "distribution": 50.0,
            },
            "support_by_action": {},
        }

    probs = action_probs[mask].astype(np.float64).copy()
    y = targets[mask].astype(np.int64)

    if not include_bc:
        probs[:, ACTION_TO_ID["BC"]] = 0.0

    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1) / max(math.log(probs.shape[1]), 1e-8)
    target_prob = probs[np.arange(len(y)), y]

    masked = probs.copy()
    masked[np.arange(len(y)), y] = -1.0
    max_other = masked.max(axis=1)
    margin = target_prob - max_other

    per_class_alignment = []
    support_by_action: dict[str, int] = {}
    for name, idx in ACTION_TO_ID.items():
        if not include_bc and name == "BC":
            continue
        cls_mask = y == idx
        if not cls_mask.any():
            continue
        per_class_alignment.append(float(target_prob[cls_mask].mean()))
        support_by_action[name] = int(cls_mask.sum())

    alignment_score = 100.0 * _safe_mean(per_class_alignment, default=float(target_prob.mean()))
    margin_score = 100.0 * float(np.clip(((margin + 1.0) / 2.0).mean(), 0.0, 1.0))
    accuracy_score = 100.0 * float((pred == y).mean())
    confidence_score = 50.0 * float(conf.mean()) + 50.0 * float(np.clip(1.0 - entropy.mean(), 0.0, 1.0))
    distribution_score = _distribution_score(probs, y, probs.shape[1])

    ncfg = cfg.get("naturalness", {})
    key = "weights_with_bc" if include_bc else "weights_without_bc"
    weights = ncfg.get(
        key,
        {"alignment": 0.35, "margin": 0.25, "accuracy": 0.20, "confidence": 0.10, "distribution": 0.10},
    )
    raw = (
        float(weights.get("alignment", 0.0)) * alignment_score
        + float(weights.get("margin", 0.0)) * margin_score
        + float(weights.get("accuracy", 0.0)) * accuracy_score
        + float(weights.get("confidence", 0.0)) * confidence_score
        + float(weights.get("distribution", 0.0)) * distribution_score
    ) / max(
        float(weights.get("alignment", 0.0))
        + float(weights.get("margin", 0.0))
        + float(weights.get("accuracy", 0.0))
        + float(weights.get("confidence", 0.0))
        + float(weights.get("distribution", 0.0)),
        1e-8,
    )
    return {
        "raw_score": float(raw),
        "support_frames": int(mask.sum()),
        "components": {
            "alignment": float(alignment_score),
            "margin": float(margin_score),
            "accuracy": float(accuracy_score),
            "confidence": float(confidence_score),
            "distribution": float(distribution_score),
        },
        "support_by_action": support_by_action,
    }


def compute_independent_scores(
    preds: dict[str, np.ndarray],
    vad0: np.ndarray,
    vad1: np.ndarray,
    action_probs: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    total_frames = min(len(vad0), len(vad1), preds["eot"].shape[0], action_probs.shape[0])
    if total_frames <= 0:
        neutral = {"score_with_bc": 3.0, "score_without_bc": 3.0}
        return neutral, {"subscores": {}, "global_stats": {}}

    vad0 = vad0[:total_frames]
    vad1 = vad1[:total_frames]
    action_probs = action_probs[:total_frames]

    targets = _derive_support_targets(vad0, vad1, cfg)[:total_frames]
    with_bc = _support_frame_action_score(action_probs, targets, include_bc=True, cfg=cfg)
    without_bc = _support_frame_action_score(action_probs, targets, include_bc=False, cfg=cfg)

    bc_only_mask = targets == ACTION_TO_ID["BC"]
    if bc_only_mask.any():
        bc_probs = action_probs[bc_only_mask]
        bc_support_score = 100.0 * float(bc_probs[:, ACTION_TO_ID["BC"]].mean())
    else:
        bc_support_score = 100.0

    global_stats = {
        "support_frames_with_bc": float(with_bc["support_frames"]),
        "support_frames_without_bc": float(without_bc["support_frames"]),
        "support_ratio_with_bc": float(with_bc["support_frames"] / max(total_frames, 1)),
        "support_ratio_without_bc": float(without_bc["support_frames"] / max(total_frames, 1)),
        "bc_support_frames": float(int(bc_only_mask.sum())),
        "support_by_action_with_bc": with_bc["support_by_action"],
        "support_by_action_without_bc": without_bc["support_by_action"],
    }

    scores = {
        "score_with_bc": _score_100_to_5(with_bc["raw_score"]),
        "score_without_bc": _score_100_to_5(without_bc["raw_score"]),
        "score_with_bc_0_100": float(with_bc["raw_score"]),
        "score_without_bc_0_100": float(without_bc["raw_score"]),
    }
    detail = {
        "subscores": {
            "alignment_with_bc": float(with_bc["components"]["alignment"]),
            "margin_with_bc": float(with_bc["components"]["margin"]),
            "accuracy_with_bc": float(with_bc["components"]["accuracy"]),
            "confidence_with_bc": float(with_bc["components"]["confidence"]),
            "distribution_with_bc": float(with_bc["components"]["distribution"]),
            "alignment_without_bc": float(without_bc["components"]["alignment"]),
            "margin_without_bc": float(without_bc["components"]["margin"]),
            "accuracy_without_bc": float(without_bc["components"]["accuracy"]),
            "confidence_without_bc": float(without_bc["components"]["confidence"]),
            "distribution_without_bc": float(without_bc["components"]["distribution"]),
            "backchannel_support": float(bc_support_score),
        },
        "global_stats": global_stats,
    }
    return scores, detail


def score_sample(
    row: dict[str, Any],
    meta: dict[str, Any],
    preds: dict[str, np.ndarray],
    vad0: np.ndarray,
    vad1: np.ndarray,
    action_probs: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    scores, detail = compute_independent_scores(preds, vad0, vad1, action_probs, cfg)
    source_natural_stem = str(meta.get("source_natural_stem", ""))
    augmentation_type = str(meta.get("augmentation_type", "unknown"))
    return {
        "session_id": row["session_id"],
        "audio_path": row.get("audio_path"),
        "json_path": row.get("json_path"),
        "augmentation_type": augmentation_type,
        "source_natural_stem": source_natural_stem,
        "score_with_bc": float(scores["score_with_bc"]),
        "score_without_bc": float(scores["score_without_bc"]),
        "score_with_bc_0_100": float(scores["score_with_bc_0_100"]),
        "score_without_bc_0_100": float(scores["score_without_bc_0_100"]),
        "subscores": detail["subscores"],
        "global_stats": detail["global_stats"],
    }


def evaluate_manifest_rows(
    rows: list[dict[str, Any]],
    predictor: dict[str, Any],
    probe: Any,
    cfg: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(rows, start=1):
        meta = _load_json(row["json_path"]) if row.get("json_path") else {}
        preds = run_audio_session_inference(predictor, row, cfg, device)
        vad0, vad1 = compute_local_vad_from_row(row, cfg)
        local_probs = _preds_to_local_probs(preds)
        action_probs = action_proba_from_probe(local_probs, probe, valid_mask=np.ones(preds["eot"].shape[0], dtype=bool))
        res = score_sample(row, meta, preds, vad0, vad1, action_probs, cfg)
        out.append(res)
        if i % 10 == 0 or i == len(rows):
            print(
                f"[{i:3d}/{len(rows)}] {row['session_id']} "
                f"with_bc={res['score_with_bc']:.2f} no_bc={res['score_without_bc']:.2f}"
            )
    return out


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_type[str(r.get("augmentation_type", "unknown"))].append(r)

    summary = {
        "num_samples": len(results),
        "overall_with_bc_mean": _safe_mean([r["score_with_bc"] for r in results]),
        "overall_without_bc_mean": _safe_mean([r["score_without_bc"] for r in results]),
        "overall_with_bc_0_100_mean": _safe_mean([r["score_with_bc_0_100"] for r in results]),
        "overall_without_bc_0_100_mean": _safe_mean([r["score_without_bc_0_100"] for r in results]),
        "by_type": {},
    }
    for k, vals in by_type.items():
        summary["by_type"][k] = {
            "n": len(vals),
            "with_bc_mean": _safe_mean([r["score_with_bc"] for r in vals]),
            "without_bc_mean": _safe_mean([r["score_without_bc"] for r in vals]),
            "alignment_with_bc_mean": _safe_mean([r["subscores"]["alignment_with_bc"] for r in vals]),
            "margin_with_bc_mean": _safe_mean([r["subscores"]["margin_with_bc"] for r in vals]),
            "accuracy_with_bc_mean": _safe_mean([r["subscores"]["accuracy_with_bc"] for r in vals]),
            "confidence_with_bc_mean": _safe_mean([r["subscores"]["confidence_with_bc"] for r in vals]),
            "distribution_with_bc_mean": _safe_mean([r["subscores"]["distribution_with_bc"] for r in vals]),
            "alignment_without_bc_mean": _safe_mean([r["subscores"]["alignment_without_bc"] for r in vals]),
            "margin_without_bc_mean": _safe_mean([r["subscores"]["margin_without_bc"] for r in vals]),
            "accuracy_without_bc_mean": _safe_mean([r["subscores"]["accuracy_without_bc"] for r in vals]),
            "confidence_without_bc_mean": _safe_mean([r["subscores"]["confidence_without_bc"] for r in vals]),
            "distribution_without_bc_mean": _safe_mean([r["subscores"]["distribution_without_bc"] for r in vals]),
            "backchannel_support_mean": _safe_mean([r["subscores"]["backchannel_support"] for r in vals]),
        }
    return summary


def pairwise_summary(natural_results: list[dict[str, Any]], unnatural_results: list[dict[str, Any]]) -> dict[str, Any]:
    natural_map = {str(r["session_id"]): r for r in natural_results}
    with_bc = []
    without_bc = []
    pair_rows = []
    for ur in unnatural_results:
        src = str(ur.get("source_natural_stem") or "")
        if not src or src not in natural_map:
            continue
        nr = natural_map[src]
        with_bc.append(float(nr["score_with_bc"]) > float(ur["score_with_bc"]))
        without_bc.append(float(nr["score_without_bc"]) > float(ur["score_without_bc"]))
        pair_rows.append(
            {
                "source_natural_stem": src,
                "unnatural_session_id": ur["session_id"],
                "augmentation_type": ur.get("augmentation_type", ""),
                "natural_with_bc": float(nr["score_with_bc"]),
                "unnatural_with_bc": float(ur["score_with_bc"]),
                "natural_without_bc": float(nr["score_without_bc"]),
                "unnatural_without_bc": float(ur["score_without_bc"]),
            }
        )
    return {
        "num_pairs": len(pair_rows),
        "with_bc_pairwise_accuracy": _safe_mean([100.0 if x else 0.0 for x in with_bc], default=0.0),
        "without_bc_pairwise_accuracy": _safe_mean([100.0 if x else 0.0 for x in without_bc], default=0.0),
        "pairs": pair_rows,
    }


def run_naturalness_evaluation(cfg: dict[str, Any], *, device: torch.device, stage2_ckpt: str | None = None) -> dict[str, Any]:
    cfg = apply_runtime_overrides(cfg)
    predictor = build_predictor(cfg, device, stage2_ckpt)

    probe_path = _get_probe_path(cfg)
    if not probe_path:
        raise ValueError("Missing action probe path. Set naturalness.action_probe_path or probe.path.")
    probe = joblib.load(probe_path)

    natural_rows = load_manifest(cfg["data"]["natural_manifest"])
    unnatural_rows = load_manifest(cfg["data"]["unnatural_manifest"])

    print("=" * 80)
    print("Scoring natural set with independent actions+VAD naturalness")
    print("=" * 80)
    natural_results = evaluate_manifest_rows(natural_rows, predictor, probe, cfg, device)

    print("=" * 80)
    print("Scoring unnatural set with independent actions+VAD naturalness")
    print("=" * 80)
    unnatural_results = evaluate_manifest_rows(unnatural_rows, predictor, probe, cfg, device)

    all_results = natural_results + unnatural_results
    return {
        "algorithm": {
            "name": "support_frame_action_naturalness",
            "description": (
                "Independent naturalness score computed only on DualTurn support frames derived from local VAD. "
                "For each support frame, the scorer compares action probabilities against the VAD-derived DualTurn "
                "action target, using alignment, margin, accuracy, confidence, and distribution. Reports both a "
                "turn-taking-only score without backchannel and a score that includes backchannel frames."
            ),
            "actions": ACTION_NAMES,
            "probe_path": str(probe_path),
        },
        "natural_results": natural_results,
        "unnatural_results": unnatural_results,
        "summary": {
            "natural": summarise(natural_results),
            "unnatural": summarise(unnatural_results),
            "all": summarise(all_results),
            "pairwise": pairwise_summary(natural_results, unnatural_results),
        },
    }


def save_naturalness_results(payload: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "naturalness_results.json"
    csv_path = out_dir / "naturalness_scores.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rows = []
    for bucket in ["natural_results", "unnatural_results"]:
        for r in payload[bucket]:
            rows.append(
                {
                    "session_id": r["session_id"],
                    "augmentation_type": r["augmentation_type"],
                    "source_natural_stem": r.get("source_natural_stem", ""),
                    "score_with_bc": r["score_with_bc"],
                    "score_without_bc": r["score_without_bc"],
                    "score_with_bc_0_100": r["score_with_bc_0_100"],
                    "score_without_bc_0_100": r["score_without_bc_0_100"],
                    "alignment_with_bc_score": r["subscores"]["alignment_with_bc"],
                    "margin_with_bc_score": r["subscores"]["margin_with_bc"],
                    "accuracy_with_bc_score": r["subscores"]["accuracy_with_bc"],
                    "confidence_with_bc_score": r["subscores"]["confidence_with_bc"],
                    "distribution_with_bc_score": r["subscores"]["distribution_with_bc"],
                    "alignment_without_bc_score": r["subscores"]["alignment_without_bc"],
                    "margin_without_bc_score": r["subscores"]["margin_without_bc"],
                    "accuracy_without_bc_score": r["subscores"]["accuracy_without_bc"],
                    "confidence_without_bc_score": r["subscores"]["confidence_without_bc"],
                    "distribution_without_bc_score": r["subscores"]["distribution_without_bc"],
                    "backchannel_support_score": r["subscores"]["backchannel_support"],
                    "audio_path": r["audio_path"],
                    "json_path": r["json_path"],
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else [
                "session_id",
                "augmentation_type",
                "source_natural_stem",
                "score_with_bc",
                "score_without_bc",
                "score_with_bc_0_100",
                "score_without_bc_0_100",
                "alignment_with_bc_score",
                "margin_with_bc_score",
                "accuracy_with_bc_score",
                "confidence_with_bc_score",
                "distribution_with_bc_score",
                "alignment_without_bc_score",
                "margin_without_bc_score",
                "accuracy_without_bc_score",
                "confidence_without_bc_score",
                "distribution_without_bc_score",
                "backchannel_support_score",
                "audio_path",
                "json_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path
