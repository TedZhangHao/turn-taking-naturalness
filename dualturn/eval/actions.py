from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

from dualturn.data.labels import ACTION_TO_ID, NO_ACTION


FEATURE_KEYS = [
    "u_eot", "u_hold", "u_bot", "u_bc", "u_vad", "u_fvad",
    "a_eot", "a_hold", "a_bot", "a_bc", "a_vad", "a_fvad",
]


def probs_to_feature_matrix(probs: dict[str, torch.Tensor]) -> np.ndarray:
    cols = []
    for key in FEATURE_KEYS:
        x = probs[key].detach().float().cpu().numpy()
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        else:
            x = x.reshape(-1, 1)
        cols.append(x)
    return np.concatenate(cols, axis=1)


def heuristic_actions_from_probs(probs: dict[str, torch.Tensor]) -> np.ndarray:
    ue = probs["u_eot"].detach().float().cpu().numpy().reshape(-1)
    uh = probs["u_hold"].detach().float().cpu().numpy().reshape(-1)
    ub = probs["u_bot"].detach().float().cpu().numpy().reshape(-1)
    uv = probs["u_vad"].detach().float().cpu().numpy().reshape(-1)
    ufv = probs["u_fvad"].detach().float().cpu().numpy().reshape(-1, probs["u_fvad"].shape[-1])

    av = probs["a_vad"].detach().float().cpu().numpy().reshape(-1)
    ab = probs["a_bot"].detach().float().cpu().numpy().reshape(-1)
    abc = probs["a_bc"].detach().float().cpu().numpy().reshape(-1)

    pred = np.full_like(ue, fill_value=NO_ACTION, dtype=np.int64)

    pred[(abc > 0.5) & (uv > 0.5)] = ACTION_TO_ID["BC"]

    st_mask = (ue > 0.5) & (ab > 0.5) & (pred == NO_ACTION)
    pred[st_mask] = ACTION_TO_ID["ST"]

    cl_mask = (uh > 0.5) & (ufv[:, -1] > 0.2) & (pred == NO_ACTION)
    pred[cl_mask] = ACTION_TO_ID["CL"]

    overlap = (ub > 0.5) & (av > 0.5) & (pred == NO_ACTION)
    long_overlap = overlap & (ufv[:, -1] > 0.2)
    short_overlap = overlap & ~long_overlap
    pred[long_overlap] = ACTION_TO_ID["SL"]
    pred[short_overlap] = ACTION_TO_ID["CT"]

    return pred


def weighted_f1_from_flat_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != NO_ACTION
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(y_true[mask], y_pred[mask], average="weighted"))


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    mask = y_true != NO_ACTION
    if mask.sum() == 0:
        return "No labeled action frames found."
    return classification_report(y_true[mask], y_pred[mask], digits=4)