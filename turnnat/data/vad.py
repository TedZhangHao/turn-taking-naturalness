from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def frame_audio(audio: torch.Tensor, frame_len: int) -> torch.Tensor:
    n = audio.shape[-1]
    pad = (frame_len - n % frame_len) % frame_len
    if pad > 0:
        audio = F.pad(audio, (0, pad))
    return audio.view(-1, frame_len)


def rms_vad(audio: torch.Tensor, frame_len: int, threshold: float) -> torch.Tensor:
    frames = frame_audio(audio, frame_len)
    rms = torch.sqrt(torch.clamp(frames.pow(2).mean(dim=-1), min=1e-8))
    return (rms >= threshold).float()


def _segments(vad: torch.Tensor, valid_mask: torch.Tensor | None = None) -> list[tuple[int, int]]:
    segs = []
    active = False
    start = 0
    T = int(vad.numel())

    for i in range(T):
        if valid_mask is not None and valid_mask[i] < 0.5:
            break

        v = float(vad[i])
        if v >= 0.5 and not active:
            start = i
            active = True
        elif v < 0.5 and active:
            segs.append((start, i - 1))
            active = False

    if active:
        end = T - 1
        if valid_mask is not None:
            valid_idxs = torch.nonzero(valid_mask > 0.5).reshape(-1)
            if valid_idxs.numel() > 0:
                end = int(valid_idxs[-1].item())
        segs.append((start, end))

    return segs


def gaussian_smooth_events(events: torch.Tensor, sigma_before: float = 3.0, sigma_after: float = 1.0) -> torch.Tensor:
    out = torch.zeros_like(events)
    idxs = torch.nonzero(events > 0.5).reshape(-1)
    T = events.numel()

    for idx in idxs.tolist():
        left = max(0, idx - 12)
        right = min(T, idx + 8)
        for t in range(left, right):
            sigma = sigma_before if t <= idx else sigma_after
            w = math.exp(-0.5 * ((t - idx) / sigma) ** 2)
            out[t] = max(float(out[t]), w)

    return out


def derive_signals(vad: torch.Tensor, frame_valid_mask: torch.Tensor, frame_hz: float = 12.5) -> dict[str, torch.Tensor]:
    T = vad.shape[-1]
    lookahead_4s = int(round(4.0 * frame_hz))
    one_sec = int(round(1.0 * frame_hz))

    raw = {
        "raw_eot": torch.zeros(2, T),
        "raw_hold": torch.zeros(2, T),
        "raw_bot": torch.zeros(2, T),
        "raw_bc": torch.zeros(2, T),
    }

    out = {
        "eot": torch.zeros(2, T),
        "hold": torch.zeros(2, T),
        "bot": torch.zeros(2, T),
        "bc": torch.zeros(2, T),
        "vad": vad.clone(),
        "fvad": torch.zeros(2, T, 4),
        **raw,
        "frame_valid_mask": frame_valid_mask.clone(),
    }

    segs = [_segments(vad[ch], frame_valid_mask) for ch in range(2)]

    for ch in range(2):
        other = 1 - ch
        own_segs = segs[ch]
        other_segs = segs[other]

        timeline = []
        for s, e in own_segs:
            timeline.append((s, e, ch))
        for s, e in other_segs:
            timeline.append((s, e, other))
        timeline.sort(key=lambda x: x[0])

        # BOT: onset >=1s following the other speaker
        for s, e in own_segs:
            dur = e - s + 1
            if dur >= one_sec:
                prev = None
                for ss, ee, cc in timeline:
                    if ss >= s:
                        break
                    prev = (ss, ee, cc)
                if prev is not None and prev[2] == other:
                    raw["raw_bot"][ch, s] = 1.0

        # BC: isolated utterance <=1s with >=1s silence before/after on same channel
        for i, (s, e) in enumerate(own_segs):
            dur = e - s + 1
            prev_end = own_segs[i - 1][1] if i > 0 else -10**9
            next_start = own_segs[i + 1][0] if i + 1 < len(own_segs) else 10**9

            if dur <= one_sec and (s - prev_end - 1) >= one_sec and (next_start - e - 1) >= one_sec:
                raw["raw_bc"][ch, s] = 1.0

        # EOT / HOLD at offsets
        for s, e in own_segs:
            next_other_start = None
            for so, eo in other_segs:
                if so > e:
                    next_other_start = so
                    break

            if next_other_start is not None and (next_other_start - e) <= lookahead_4s:
                raw["raw_eot"][ch, e] = 1.0
            else:
                raw["raw_hold"][ch, e] = 1.0

    for name, raw_name in [("eot", "raw_eot"), ("hold", "raw_hold"), ("bot", "raw_bot"), ("bc", "raw_bc")]:
        for ch in range(2):
            out[name][ch] = gaussian_smooth_events(raw[raw_name][ch])

    horizons = [
        int(round(0.24 * frame_hz)),
        int(round(0.48 * frame_hz)),
        int(round(0.96 * frame_hz)),
        int(round(2.0 * frame_hz)),
    ]
    prev = 0
    valid_len = int(frame_valid_mask.sum().item())

    for hi, h in enumerate(horizons):
        for t in range(valid_len):
            a = min(valid_len, t + prev)
            b = min(valid_len, t + h)
            if b <= a:
                continue
            for ch in range(2):
                out["fvad"][ch, t, hi] = vad[ch, a:b].float().mean()
        prev = h

    invalid = frame_valid_mask < 0.5
    for name in ["eot", "hold", "bot", "bc", "raw_eot", "raw_hold", "raw_bot", "raw_bc", "vad"]:
        out[name][:, invalid] = 0.0
    out["fvad"][:, invalid] = 0.0
    return out