from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import torch

from dualturn.data.manifest import load_manifest
from dualturn.eval.actions import probs_to_feature_matrix
from dualturn.eval.official_event_eval import build_predictor, run_audio_session_inference


ACTION_NAMES = ["ST", "CL", "SL", "CT", "BC"]
ACTION_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}
NON_BC_ACTIONS = ["ST", "CL", "SL", "CT"]


def _call_flexible(fn: Callable, /, **kwargs):
    """
    Call a function using only the keyword arguments that exist in its signature.
    This makes this file tolerant to small differences in official_event_eval.py.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    usable = {k: v for k, v in kwargs.items() if k in params}
    return fn(**usable)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unwrap_inference_result(result: Any) -> dict[str, Any]:
    """
    Find the probability/prediction dictionary returned by run_audio_session_inference.

    Your current official_event_eval.run_audio_session_inference returns:
      preds["eot"], preds["hold"], preds["bot"], preds["bc"], preds["vad"] with shape [T,2]
      preds["fvad"] with shape [T,8]

    Older paths may return u_eot/a_eot/... or nested {"probs": ...}. This supports both.
    """
    if isinstance(result, (tuple, list)):
        for obj in result:
            if isinstance(obj, dict):
                try:
                    return _unwrap_inference_result(obj)
                except Exception:
                    pass
        raise RuntimeError("Could not find probability dict inside tuple/list inference result.")

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected inference result type: {type(result)}")

    for key in ["probs", "signal_probs", "preds", "predictions", "outputs", "result", "raw"]:
        if key in result and isinstance(result[key], dict):
            try:
                return _unwrap_inference_result(result[key])
            except Exception:
                pass

    flat_keys = set(result.keys())

    # Current official_event_eval.py returns these keys directly.
    if {"eot", "hold", "bot", "bc", "vad", "fvad"} & flat_keys:
        return result

    # Older local-prob layout.
    if {"u_eot", "a_eot", "u_vad", "a_vad", "vad_probs", "fvad_probs"} & flat_keys:
        return result

    for value in result.values():
        if isinstance(value, dict):
            try:
                return _unwrap_inference_result(value)
            except Exception:
                pass

    debug = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            debug[k] = f"Tensor{tuple(v.shape)}"
        elif isinstance(v, np.ndarray):
            debug[k] = f"ndarray{v.shape}"
        elif isinstance(v, dict):
            debug[k] = f"dict keys={list(v.keys())[:20]}"
        else:
            debug[k] = str(type(v))
    raise RuntimeError(
        "Could not standardize inference result. "
        f"Expected eot/hold/bot/bc/vad/fvad or u_eot/a_eot keys. Got: {debug}"
    )


def _standardize_probs(probs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    Convert official_event_eval preds into the layout expected by both:
      1) legacy probs_to_feature_matrix: u_eot/u_hold/.../a_fvad
      2) official_event_probe_v1 bundle: we can reconstruct eot/hold/... from u/a keys

    Input supported:
      eot/hold/bot/bc/vad: [T,2], fvad: [T,8]
      or u_eot/a_eot/... directly.
    """
    raw: dict[str, torch.Tensor] = {}
    for k, v in probs.items():
        if isinstance(v, torch.Tensor):
            raw[k] = v.detach().cpu().float()
        elif isinstance(v, np.ndarray):
            raw[k] = torch.from_numpy(v).detach().cpu().float()
        elif isinstance(v, (list, tuple)):
            try:
                raw[k] = torch.tensor(v, dtype=torch.float32)
            except Exception:
                pass

    out: dict[str, torch.Tensor] = {}

    # Case A: official_event_eval layout: eot/hold/bot/bc/vad = [T,2], fvad=[T,8].
    for src_key, name in [("eot", "eot"), ("hold", "hold"), ("bot", "bot"), ("bc", "bc"), ("vad", "vad")]:
        if src_key in raw:
            x = raw[src_key]
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2 and x.shape[-1] >= 2:
                out[f"u_{name}"] = x[:, 0]
                out[f"a_{name}"] = x[:, 1]
            elif x.ndim == 2 and x.shape[0] >= 2:
                out[f"u_{name}"] = x[0]
                out[f"a_{name}"] = x[1]

    if "fvad" in raw:
        fvad = raw["fvad"]
        if fvad.ndim == 3 and fvad.shape[0] == 1:
            fvad = fvad[0]
        if fvad.ndim == 2 and fvad.shape[-1] >= 8:
            out["u_fvad"] = fvad[:, 0:4]
            out["a_fvad"] = fvad[:, 4:8]
        elif fvad.ndim == 2 and fvad.shape[0] >= 8:
            out["u_fvad"] = fvad[0:4].T
            out["a_fvad"] = fvad[4:8].T

    # Case B: wrapper layout: vad_probs/fvad_probs/eot_probs/... = [B,T,2] or [T,2].
    if "vad_probs" in raw and ("u_vad" not in out or "a_vad" not in out):
        vad = raw["vad_probs"]
        if vad.ndim == 3 and vad.shape[0] == 1:
            vad = vad[0]
        if vad.ndim == 2 and vad.shape[-1] >= 2:
            out["u_vad"] = vad[:, 0]
            out["a_vad"] = vad[:, 1]

    if "fvad_probs" in raw and ("u_fvad" not in out or "a_fvad" not in out):
        fvad = raw["fvad_probs"]
        if fvad.ndim == 3 and fvad.shape[0] == 1:
            fvad = fvad[0]
        if fvad.ndim == 2 and fvad.shape[-1] >= 8:
            out["u_fvad"] = fvad[:, 0:4]
            out["a_fvad"] = fvad[:, 4:8]

    for src_key, name in [("eot_probs", "eot"), ("hold_probs", "hold"), ("bot_probs", "bot"), ("bc_probs", "bc")]:
        if src_key in raw:
            x = raw[src_key]
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2 and x.shape[-1] >= 2:
                out[f"u_{name}"] = x[:, 0]
                out[f"a_{name}"] = x[:, 1]

    # Case C: already local layout.
    for k, v in raw.items():
        if k.startswith(("u_", "a_")) or k == "frame_valid_mask":
            x = v
            if x.ndim >= 2 and x.shape[0] == 1 and k not in {"u_fvad", "a_fvad"}:
                x = x[0]
            elif x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            out[k] = x

    expected_scalar = [
        "u_eot", "u_hold", "u_bot", "u_bc", "u_vad",
        "a_eot", "a_hold", "a_bot", "a_bc", "a_vad",
    ]
    expected_dense = ["u_fvad", "a_fvad"]

    T = None
    for k in expected_scalar:
        if k in out:
            T = int(out[k].shape[-1])
            break
    if T is None:
        for k in expected_dense:
            if k in out:
                T = int(out[k].shape[-2] if out[k].ndim >= 2 else out[k].shape[0])
                break
    if T is None:
        raise RuntimeError(f"Could not infer frame length from parsed keys: {sorted(out.keys())}; raw keys={sorted(raw.keys())}")

    normalized: dict[str, torch.Tensor] = {}
    for k in expected_scalar:
        x = out.get(k, torch.zeros(T, dtype=torch.float32)).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        normalized[k] = x[:, :T]

    for k in expected_dense:
        x = out.get(k, torch.zeros(T, 4, dtype=torch.float32)).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        normalized[k] = x[:, :T, :4]

    if "frame_valid_mask" in out:
        mask = out["frame_valid_mask"].float()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        normalized["frame_valid_mask"] = mask[:, :T]
    else:
        normalized["frame_valid_mask"] = torch.ones(1, T, dtype=torch.float32)

    T_min = min(normalized[k].shape[1] for k in normalized if normalized[k].ndim >= 2)
    for k in list(normalized.keys()):
        if normalized[k].ndim == 2:
            normalized[k] = normalized[k][:, :T_min]
        elif normalized[k].ndim == 3:
            normalized[k] = normalized[k][:, :T_min, :]

    return normalized


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _duration_from_row_or_meta(row: dict[str, Any], meta: dict[str, Any], frame_count: int, frame_hz: float) -> float:
    for key in ["duration_sec", "duration", "duration_s"]:
        if row.get(key):
            try:
                return float(row[key])
            except Exception:
                pass
        if meta.get(key):
            try:
                return float(meta[key])
            except Exception:
                pass
    return max(frame_count / max(frame_hz, 1e-8), 1e-8)


def _local_probs_to_pred_mats(probs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Reconstruct official_event_eval-style pred matrices from u_*/a_* tensors."""
    def arr1(k: str) -> np.ndarray:
        x = probs[k].detach().float().cpu().numpy()
        if x.ndim == 2:
            x = x[0]
        return x.astype(np.float32)

    def arrf(k: str) -> np.ndarray:
        x = probs[k].detach().float().cpu().numpy()
        if x.ndim == 3:
            x = x[0]
        return x.astype(np.float32)

    u_eot, a_eot = arr1("u_eot"), arr1("a_eot")
    T = min(len(u_eot), len(a_eot))

    preds = {
        "eot": np.stack([arr1("u_eot")[:T], arr1("a_eot")[:T]], axis=1),
        "hold": np.stack([arr1("u_hold")[:T], arr1("a_hold")[:T]], axis=1),
        "bot": np.stack([arr1("u_bot")[:T], arr1("a_bot")[:T]], axis=1),
        "bc": np.stack([arr1("u_bc")[:T], arr1("a_bc")[:T]], axis=1),
        "vad": np.stack([arr1("u_vad")[:T], arr1("a_vad")[:T]], axis=1),
        "fvad": np.concatenate([arrf("u_fvad")[:T, :4], arrf("a_fvad")[:T, :4]], axis=1),
    }
    return preds


def _official_signal_matrix_from_preds(preds: dict[str, np.ndarray], user_ch: int, agent_ch: int) -> np.ndarray:
    """
    Build [T,16] features in the exact SIGNAL_NAMES order used by official_event_eval.py.
    This mirrors extract_signals_at_frame but vectorized for every frame.
    """
    T = int(preds["eot"].shape[0])
    fvad = preds["fvad"]
    out = np.zeros((T, 16), dtype=np.float32)

    out[:, 0] = preds["eot"][:, user_ch]
    out[:, 1] = preds["eot"][:, agent_ch]
    out[:, 2] = preds["hold"][:, user_ch]
    out[:, 3] = preds["hold"][:, agent_ch]
    out[:, 4] = preds["bot"][:, user_ch]
    out[:, 5] = preds["bot"][:, agent_ch]
    out[:, 6] = preds["bc"][:, user_ch]
    out[:, 7] = preds["bc"][:, agent_ch]
    out[:, 8] = preds["vad"][:, user_ch]
    out[:, 9] = preds["vad"][:, agent_ch]
    out[:, 10] = fvad[:, user_ch * 4 : user_ch * 4 + 2].max(axis=1)
    out[:, 11] = fvad[:, user_ch * 4 + 2 : user_ch * 4 + 4].max(axis=1)
    out[:, 12] = fvad[:, agent_ch * 4 : agent_ch * 4 + 2].max(axis=1)
    out[:, 13] = fvad[:, agent_ch * 4 + 2 : agent_ch * 4 + 4].max(axis=1)
    out[:, 14] = out[:, 0] - out[:, 2]
    out[:, 15] = out[:, 1] - out[:, 3]
    return out


def _predict_with_official_probe_bundle(probs: dict[str, torch.Tensor], bundle: dict[str, Any], valid_mask: np.ndarray) -> np.ndarray:
    """
    Use official_event_probe_v1 bundle. The bundle stores fitted_5class['LR-all'] =
    {scaler, lr, feat_idx, feat_names}; it is not a bare LogisticRegression.

    We compute action probabilities in both channel orientations and take per-class max.
    This lets the global naturalness score count possible actions from either side.
    """
    if bundle.get("format") != "official_event_probe_v1":
        raise RuntimeError(f"Unsupported probe bundle format: {bundle.get('format')}")

    fit = bundle["fitted_5class"].get("LR-all")
    if fit is None:
        # fallback to strongest available subset if LR-all is missing
        fit = next(iter(bundle["fitted_5class"].values()))

    preds = _local_probs_to_pred_mats(probs)
    T = min(preds["eot"].shape[0], valid_mask.shape[0])
    valid_mask = valid_mask[:T]

    all_orient_probs = []
    for user_ch, agent_ch in [(0, 1), (1, 0)]:
        X_all = _official_signal_matrix_from_preds(preds, user_ch=user_ch, agent_ch=agent_ch)[:T]
        X_sub = X_all[:, fit["feat_idx"]]
        X_scaled = fit["scaler"].transform(X_sub)
        raw = fit["lr"].predict_proba(X_scaled)

        out = np.zeros((raw.shape[0], len(ACTION_NAMES)), dtype=np.float32)
        classes = list(getattr(fit["lr"], "classes_", range(raw.shape[1])))
        for src_i, cls in enumerate(classes):
            cls_i = int(cls)
            if 0 <= cls_i < len(ACTION_NAMES):
                out[:, cls_i] = raw[:, src_i]
        all_orient_probs.append(out)

    action_probs = np.maximum(all_orient_probs[0], all_orient_probs[1])
    action_probs[~valid_mask.astype(bool)] = 0.0
    return action_probs


def action_proba_from_probe(
    probs: dict[str, torch.Tensor],
    probe: Any,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert DualTurn signal probabilities to soft action probabilities.

    Supports two probe artifacts:
      A) official_event_probe_v1 bundle from your current fit_action_probe.py
      B) bare LogisticRegression from the older legacy fit_action_probe.py

    Returns [T,5], columns = ST, CL, SL, CT, BC.
    """
    if isinstance(probe, dict) and probe.get("format") == "official_event_probe_v1":
        return _predict_with_official_probe_bundle(probs, probe, valid_mask)

    # Legacy direct LogisticRegression path.
    X = probs_to_feature_matrix(probs)
    T = valid_mask.shape[0]
    if X.shape[0] != T:
        n = min(X.shape[0], T)
        X = X[:n]
        valid_mask = valid_mask[:n]
        T = n

    if not hasattr(probe, "predict_proba"):
        raise RuntimeError(
            "Action probe must either be official_event_probe_v1 bundle or support predict_proba."
        )

    raw = probe.predict_proba(X)
    out = np.zeros((raw.shape[0], len(ACTION_NAMES)), dtype=np.float32)
    classes = list(getattr(probe, "classes_", range(raw.shape[1])))
    for src_i, cls in enumerate(classes):
        cls_i = int(cls)
        if 0 <= cls_i < len(ACTION_NAMES):
            out[:, cls_i] = raw[:, src_i]

    out[~valid_mask.astype(bool)] = 0.0
    return out


def hard_action_events(
    action_probs: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    include_bc: bool,
) -> np.ndarray:
    """
    Return event frame indices from soft action probabilities.
    Consecutive frames with same high-probability action are collapsed to one event.
    """
    if action_probs.size == 0:
        return np.asarray([], dtype=np.int64)

    probs = action_probs.copy()
    if not include_bc:
        probs[:, ACTION_TO_ID["BC"]] = 0.0

    best_id = probs.argmax(axis=1)
    best_p = probs.max(axis=1)

    active = (best_p >= threshold) & valid_mask.astype(bool)
    idxs = np.where(active)[0]
    if idxs.size == 0:
        return idxs.astype(np.int64)

    keep = [int(idxs[0])]
    last_idx = int(idxs[0])
    last_action = int(best_id[last_idx])

    for idx in idxs[1:]:
        idx = int(idx)
        action = int(best_id[idx])
        if idx != last_idx + 1 or action != last_action:
            keep.append(idx)
        last_idx = idx
        last_action = action

    return np.asarray(keep, dtype=np.int64)


def entropy_from_action_probs(action_probs: np.ndarray, include_bc: bool) -> np.ndarray:
    p = action_probs.copy().astype(np.float64)
    if not include_bc:
        p[:, ACTION_TO_ID["BC"]] = 0.0

    denom = p.sum(axis=1, keepdims=True)
    p = np.divide(p, np.maximum(denom, 1e-12))
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)


def extract_global_action_features(
    action_probs: np.ndarray,
    valid_mask: np.ndarray,
    duration_sec: float,
    frame_hz: float,
    hard_event_threshold: float,
) -> dict[str, float]:
    valid = valid_mask.astype(bool)
    if valid.sum() == 0:
        valid[:] = True

    p = action_probs[valid]
    dur = max(float(duration_sec), 1e-8)

    feats: dict[str, float] = {}
    for i, name in enumerate(ACTION_NAMES):
        mass = float(p[:, i].sum())
        feats[f"{name.lower()}_mass"] = mass
        feats[f"{name.lower()}_rate"] = mass / dur
        feats[f"{name.lower()}_mean_prob"] = float(p[:, i].mean()) if p.size else 0.0
        feats[f"{name.lower()}_p90_prob"] = float(np.percentile(p[:, i], 90)) if p.size else 0.0

    non_bc_ids = [ACTION_TO_ID[x] for x in NON_BC_ACTIONS]
    feats["all_action_mass"] = float(p.sum())
    feats["all_action_rate"] = feats["all_action_mass"] / dur
    feats["non_bc_action_mass"] = float(p[:, non_bc_ids].sum())
    feats["non_bc_action_rate"] = feats["non_bc_action_mass"] / dur

    # Timing features from collapsed high-confidence events.
    for include_bc, prefix in [(True, "with_bc"), (False, "without_bc")]:
        ev = hard_action_events(
            action_probs=action_probs,
            valid_mask=valid_mask,
            threshold=hard_event_threshold,
            include_bc=include_bc,
        )
        feats[f"{prefix}_event_count"] = float(len(ev))
        feats[f"{prefix}_event_rate"] = float(len(ev)) / dur

        if len(ev) >= 2:
            gaps_sec = np.diff(ev) / max(frame_hz, 1e-8)
            feats[f"{prefix}_event_gap_mean"] = float(np.mean(gaps_sec))
            feats[f"{prefix}_event_gap_std"] = float(np.std(gaps_sec))
            feats[f"{prefix}_event_gap_p90"] = float(np.percentile(gaps_sec, 90))
        else:
            feats[f"{prefix}_event_gap_mean"] = dur
            feats[f"{prefix}_event_gap_std"] = 0.0
            feats[f"{prefix}_event_gap_p90"] = dur

    # Confidence / entropy features.
    ent_with = entropy_from_action_probs(action_probs[valid], include_bc=True)
    ent_without = entropy_from_action_probs(action_probs[valid], include_bc=False)

    conf_with = p.max(axis=1) if p.size else np.asarray([0.0])
    p_no_bc = p[:, non_bc_ids] if p.size else np.zeros((1, 4))
    conf_without = p_no_bc.max(axis=1)

    feats["with_bc_entropy_mean"] = float(ent_with.mean()) if ent_with.size else 0.0
    feats["with_bc_entropy_p90"] = float(np.percentile(ent_with, 90)) if ent_with.size else 0.0
    feats["with_bc_confidence_mean"] = float(conf_with.mean()) if conf_with.size else 0.0
    feats["with_bc_confidence_p90"] = float(np.percentile(conf_with, 90)) if conf_with.size else 0.0

    feats["without_bc_entropy_mean"] = float(ent_without.mean()) if ent_without.size else 0.0
    feats["without_bc_entropy_p90"] = float(np.percentile(ent_without, 90)) if ent_without.size else 0.0
    feats["without_bc_confidence_mean"] = float(conf_without.mean()) if conf_without.size else 0.0
    feats["without_bc_confidence_p90"] = float(np.percentile(conf_without, 90)) if conf_without.size else 0.0

    return feats


def build_baseline(feature_rows: list[dict[str, Any]], feature_names: list[str], robust: bool) -> dict[str, dict[str, float]]:
    baseline: dict[str, dict[str, float]] = {}
    for name in feature_names:
        vals = []
        for row in feature_rows:
            v = row.get(name)
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)
            except Exception:
                pass

        if not vals:
            baseline[name] = {
                "center": 0.0,
                "scale": 1.0,
                "mean": 0.0,
                "std": 1.0,
                "median": 0.0,
                "mad": 1.0,
            }
            continue

        arr = np.asarray(vals, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))

        if robust:
            center = median
            scale = 1.4826 * mad
            if scale < 1e-8:
                scale = std
        else:
            center = mean
            scale = std

        if scale < 1e-8:
            scale = 1.0

        baseline[name] = {
            "center": float(center),
            "scale": float(scale),
            "mean": mean,
            "std": max(std, 1e-8),
            "median": median,
            "mad": max(mad, 1e-8),
        }

    return baseline


def feature_score(value: float, stats: dict[str, float], scale: float) -> float:
    z = abs(float(value) - float(stats["center"])) / max(float(stats["scale"]), 1e-8)
    return float(100.0 * math.exp(-z / max(scale, 1e-8)))


def average_scores(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 100.0
    return float(np.mean(vals))


WITH_BC_DIST_FEATURES = [
    "st_rate", "cl_rate", "sl_rate", "ct_rate", "bc_rate", "all_action_rate",
    "st_mean_prob", "cl_mean_prob", "sl_mean_prob", "ct_mean_prob", "bc_mean_prob",
]
WITHOUT_BC_DIST_FEATURES = [
    "st_rate", "cl_rate", "sl_rate", "ct_rate", "non_bc_action_rate",
    "st_mean_prob", "cl_mean_prob", "sl_mean_prob", "ct_mean_prob",
]
WITH_BC_TIMING_FEATURES = [
    "with_bc_event_rate", "with_bc_event_gap_mean", "with_bc_event_gap_std", "with_bc_event_gap_p90",
]
WITHOUT_BC_TIMING_FEATURES = [
    "without_bc_event_rate", "without_bc_event_gap_mean", "without_bc_event_gap_std", "without_bc_event_gap_p90",
]
WITH_BC_CONF_FEATURES = [
    "with_bc_entropy_mean", "with_bc_entropy_p90",
    "with_bc_confidence_mean", "with_bc_confidence_p90",
]
WITHOUT_BC_CONF_FEATURES = [
    "without_bc_entropy_mean", "without_bc_entropy_p90",
    "without_bc_confidence_mean", "without_bc_confidence_p90",
]

ALL_BASELINE_FEATURES = sorted(
    set(
        WITH_BC_DIST_FEATURES
        + WITHOUT_BC_DIST_FEATURES
        + WITH_BC_TIMING_FEATURES
        + WITHOUT_BC_TIMING_FEATURES
        + WITH_BC_CONF_FEATURES
        + WITHOUT_BC_CONF_FEATURES
    )
)

def to_1_5(x: float) -> float:
    x = max(0.0, min(100.0, float(x)))
    return 1.0 + 4.0 * x / 100.0

def score_one_row(
    features: dict[str, Any],
    baseline: dict[str, dict[str, float]],
    cfg: dict[str, Any],
) -> dict[str, float]:
    scfg = cfg.get("action_naturalness", {})
    dist_scale = float(scfg.get("distance_score_scale", 1.5))

    weights_with = scfg.get(
        "weights_with_bc",
        {"action_dist": 0.45, "action_timing": 0.30, "action_confidence": 0.25},
    )
    weights_without = scfg.get(
        "weights_without_bc",
        {"action_dist": 0.45, "action_timing": 0.35, "action_confidence": 0.20},
    )

    def component(feature_names: list[str]) -> float:
        vals = []
        for name in feature_names:
            if name not in baseline or name not in features:
                continue
            vals.append(feature_score(float(features[name]), baseline[name], dist_scale))
        return average_scores(vals)

    with_dist = component(WITH_BC_DIST_FEATURES)
    with_timing = component(WITH_BC_TIMING_FEATURES)
    with_conf = component(WITH_BC_CONF_FEATURES)

    without_dist = component(WITHOUT_BC_DIST_FEATURES)
    without_timing = component(WITHOUT_BC_TIMING_FEATURES)
    without_conf = component(WITHOUT_BC_CONF_FEATURES)

    score_with = (
        float(weights_with.get("action_dist", 0.0)) * with_dist
        + float(weights_with.get("action_timing", 0.0)) * with_timing
        + float(weights_with.get("action_confidence", 0.0)) * with_conf
    )
    denom_with = (
        float(weights_with.get("action_dist", 0.0))
        + float(weights_with.get("action_timing", 0.0))
        + float(weights_with.get("action_confidence", 0.0))
    )
    score_with = score_with / max(denom_with, 1e-8)

    score_without = (
        float(weights_without.get("action_dist", 0.0)) * without_dist
        + float(weights_without.get("action_timing", 0.0)) * without_timing
        + float(weights_without.get("action_confidence", 0.0)) * without_conf
    )
    denom_without = (
        float(weights_without.get("action_dist", 0.0))
        + float(weights_without.get("action_timing", 0.0))
        + float(weights_without.get("action_confidence", 0.0))
    )
    score_without = score_without / max(denom_without, 1e-8)

    return {
        # 1~5 final scores
        "score_with_bc": to_1_5(score_with),
        "score_without_bc": to_1_5(score_without),

        # keep original 0~100 scores for debugging
        "score_with_bc_0_100": float(score_with),
        "score_without_bc_0_100": float(score_without),

        # component scores, also converted to 1~5
        "with_bc_action_dist": to_1_5(with_dist),
        "with_bc_action_timing": to_1_5(with_timing),
        "with_bc_action_confidence": to_1_5(with_conf),
        "without_bc_action_dist": to_1_5(without_dist),
        "without_bc_action_timing": to_1_5(without_timing),
        "without_bc_action_confidence": to_1_5(without_conf),

        # optional original component scores for debugging
        "with_bc_action_dist_0_100": float(with_dist),
        "with_bc_action_timing_0_100": float(with_timing),
        "with_bc_action_confidence_0_100": float(with_conf),
        "without_bc_action_dist_0_100": float(without_dist),
        "without_bc_action_timing_0_100": float(without_timing),
        "without_bc_action_confidence_0_100": float(without_conf),
    }

def _limit_rows(rows: list[dict[str, Any]], max_n: int) -> list[dict[str, Any]]:
    if max_n and max_n > 0:
        return rows[:max_n]
    return rows


def process_manifest_rows(
    rows: list[dict[str, Any]],
    set_name: str,
    cfg: dict[str, Any],
    device: torch.device,
    predictor: Any,
    probe: Any,
) -> list[dict[str, Any]]:
    scfg = cfg.get("action_naturalness", {})
    frame_hz = float(cfg.get("data", {}).get("frame_hz", scfg.get("frame_hz", 12.5)))
    threshold = float(scfg.get("hard_event_threshold", 0.40))

    out_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        session_id = row.get("session_id") or row.get("id") or Path(row.get("audio_path", f"row{idx}")).stem
        print(f"[{set_name} {idx}/{len(rows)}] {session_id}")

        meta = {}
        if row.get("json_path"):
            try:
                meta = _load_json(row["json_path"])
            except Exception:
                meta = {}

        raw_result = _call_flexible(
            run_audio_session_inference,
            cfg=cfg,
            config=cfg,
            row=row,
            manifest_row=row,
            predictor=predictor,
            device=device,
            audio_path=row.get("audio_path"),
            json_path=row.get("json_path"),
        )
        probs = _standardize_probs(_unwrap_inference_result(raw_result))

        valid_mask = _as_numpy(probs["frame_valid_mask"])
        if valid_mask.ndim == 2:
            valid_mask = valid_mask[0]
        valid_mask = valid_mask.astype(bool)

        action_probs = action_proba_from_probe(probs, probe, valid_mask=valid_mask)
        n = min(action_probs.shape[0], valid_mask.shape[0])
        action_probs = action_probs[:n]
        valid_mask = valid_mask[:n]

        duration_sec = _duration_from_row_or_meta(row, meta, frame_count=n, frame_hz=frame_hz)
        feats = extract_global_action_features(
            action_probs=action_probs,
            valid_mask=valid_mask,
            duration_sec=duration_sec,
            frame_hz=frame_hz,
            hard_event_threshold=threshold,
        )

        edit_meta = meta.get("edit_meta", {}) if isinstance(meta, dict) else {}
        result_row: dict[str, Any] = {
            "set": set_name,
            "id": row.get("id", session_id),
            "session_id": session_id,
            "audio_path": row.get("audio_path", ""),
            "json_path": row.get("json_path", ""),
            "duration_sec": duration_sec,
            "augmentation_type": meta.get("augmentation_type", ""),
            "edit_type": edit_meta.get("edit_type", meta.get("edit_type", "")) if isinstance(edit_meta, dict) else "",
        }
        result_row.update(feats)
        out_rows.append(result_row)

    return out_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    keys = ["score_with_bc", "score_without_bc"]
    out: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and str(r[key]) != ""]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_std"] = float(arr.std())
            out[f"{key}_median"] = float(np.median(arr))

    by_edit: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        edit = str(r.get("edit_type") or r.get("augmentation_type") or "unknown")
        by_edit.setdefault(edit, []).append(r)

    out["by_edit_type"] = {}
    for edit, sub in by_edit.items():
        vals_w = [float(r["score_with_bc"]) for r in sub if "score_with_bc" in r]
        vals_wo = [float(r["score_without_bc"]) for r in sub if "score_without_bc" in r]
        out["by_edit_type"][edit] = {
            "n": len(sub),
            "score_with_bc_mean": float(np.mean(vals_w)) if vals_w else None,
            "score_without_bc_mean": float(np.mean(vals_wo)) if vals_wo else None,
        }

    return out


def run_action_naturalness_evaluation(
    cfg: dict[str, Any],
    device: torch.device | None = None,
    probe_path: str | Path | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scfg = cfg.get("action_naturalness", {})

    natural_manifest = cfg["data"]["natural_manifest"]
    unnatural_manifest = cfg["data"].get("unnatural_manifest")

    natural_rows = load_manifest(natural_manifest)
    unnatural_rows = load_manifest(unnatural_manifest) if unnatural_manifest else []

    natural_rows = _limit_rows(natural_rows, int(scfg.get("max_natural", 0)))
    unnatural_rows = _limit_rows(unnatural_rows, int(scfg.get("max_unnatural", 0)))

    final_probe_path = (
        probe_path
        or scfg.get("probe_path")
        or cfg.get("probe", {}).get("path")
    )
    if not final_probe_path:
        raise ValueError("Missing action probe path. Pass --probe or set action_naturalness.probe_path.")
    probe = joblib.load(final_probe_path)

    predictor = _call_flexible(
        build_predictor,
        cfg=cfg,
        config=cfg,
        device=device,
        stage2_ckpt=None,
    )

    print("========== Processing natural set ==========")
    natural_feature_rows = process_manifest_rows(
        rows=natural_rows,
        set_name="natural",
        cfg=cfg,
        device=device,
        predictor=predictor,
        probe=probe,
    )

    robust = bool(scfg.get("robust_baseline", True))
    baseline = build_baseline(
        natural_feature_rows,
        feature_names=ALL_BASELINE_FEATURES,
        robust=robust,
    )

    scored_natural: list[dict[str, Any]] = []
    for r in natural_feature_rows:
        rr = dict(r)
        rr.update(score_one_row(rr, baseline, cfg))
        scored_natural.append(rr)

    print("========== Processing unnatural set ==========")
    unnatural_feature_rows = process_manifest_rows(
        rows=unnatural_rows,
        set_name="unnatural",
        cfg=cfg,
        device=device,
        predictor=predictor,
        probe=probe,
    )

    scored_unnatural: list[dict[str, Any]] = []
    for r in unnatural_feature_rows:
        rr = dict(r)
        rr.update(score_one_row(rr, baseline, cfg))
        scored_unnatural.append(rr)

    payload = {
        "config": cfg,
        "probe_path": str(final_probe_path),
        "algorithm": {
            "name": "global_action_naturalness_no_anchor",
            "description": (
                "Scores any two-channel conversation by comparing global DualTurn action "
                "distribution, action timing, and action confidence against a natural baseline. "
                "No edit anchor or edit_meta is used."
            ),
            "actions": ACTION_NAMES,
            "baseline_features": ALL_BASELINE_FEATURES,
        },
        "baseline": baseline,
        "rows": scored_natural + scored_unnatural,
        "summary": {
            "natural": summarize(scored_natural),
            "unnatural": summarize(scored_unnatural),
        },
    }
    return payload


def save_action_naturalness_results(payload: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "action_naturalness_results.json"
    csv_path = out_dir / "action_naturalness_scores.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rows = payload.get("rows", [])
    keys: list[str] = []
    seen = set()
    preferred = [
        "set", "id", "session_id", "edit_type", "augmentation_type",
        "duration_sec", "score_with_bc", "score_without_bc",
        "with_bc_action_dist", "with_bc_action_timing", "with_bc_action_confidence",
        "without_bc_action_dist", "without_bc_action_timing", "without_bc_action_confidence",
        "st_rate", "cl_rate", "sl_rate", "ct_rate", "bc_rate",
        "with_bc_event_rate", "without_bc_event_rate",
        "with_bc_entropy_mean", "without_bc_entropy_mean",
        "audio_path", "json_path",
    ]
    for k in preferred:
        keys.append(k)
        seen.add(k)

    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    return json_path, csv_path
