from __future__ import annotations

import torch

NO_ACTION = -100
ACTION_TO_ID = {"ST": 0, "CL": 1, "SL": 2, "CT": 3, "BC": 4}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


def _segments(vad: torch.Tensor, valid_mask: torch.Tensor) -> list[tuple[int, int]]:
    segs = []
    active = False
    start = 0
    T = int(valid_mask.sum().item())

    for i in range(T):
        v = float(vad[i])
        if v >= 0.5 and not active:
            start = i
            active = True
        elif v < 0.5 and active:
            segs.append((start, i - 1))
            active = False

    if active:
        segs.append((start, T - 1))
    return segs


def derive_action_targets(signals: dict[str, torch.Tensor], frame_hz: float = 12.5) -> torch.Tensor:
    valid_mask = signals["frame_valid_mask"]
    T = signals["vad"].shape[-1]
    y = torch.full((T,), NO_ACTION, dtype=torch.long)

    user_vad = signals["vad"][0]
    agent_vad = signals["vad"][1]
    user_segs = _segments(user_vad, valid_mask)
    agent_segs = _segments(agent_vad, valid_mask)

    within_4s = int(round(4.0 * frame_hz))
    within_2s = int(round(2.0 * frame_hz))
    one_sec = int(round(1.0 * frame_hz))

    # BC: agent vocalization <1s during user speech
    for s, e in agent_segs:
        dur = e - s + 1
        overlap = float(user_vad[s : e + 1].max()) > 0.5
        if dur < one_sec and overlap:
            y[s] = ACTION_TO_ID["BC"]

    # ST / CL from user offset
    for s, e in user_segs:
        next_user_start = None
        next_agent_start = None

        for su, eu in user_segs:
            if su > e:
                next_user_start = su
                break

        for sa, ea in agent_segs:
            if sa > e:
                next_agent_start = sa
                break

        candidates = []
        if next_user_start is not None and (next_user_start - e) <= within_2s:
            candidates.append(("CL", next_user_start))
        if next_agent_start is not None and (next_agent_start - e) <= within_4s:
            candidates.append(("ST", next_agent_start))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            y[e] = ACTION_TO_ID[candidates[0][0]]

    # SL / CT: overlap onset; incoming speech >1s / <1s
    for s, e in user_segs:
        if s > 0 and float(agent_vad[s - 1]) > 0.5:
            dur = e - s + 1
            if dur > one_sec:
                y[s] = ACTION_TO_ID["SL"]
            else:
                y[s] = ACTION_TO_ID["CT"]

    y[valid_mask < 0.5] = NO_ACTION
    return y