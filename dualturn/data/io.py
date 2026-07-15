from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from scipy.signal import resample_poly


def _read_audio_bytes_from_tar(tar_path: str | Path, member_name: str) -> tuple[torch.Tensor, int]:
    with tarfile.open(tar_path, 'r') as tf:
        member = tf.getmember(member_name)
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f'Could not extract {member_name} from {tar_path}')
        audio_bytes = f.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    audio = torch.from_numpy(audio.T).float()
    return audio, sr


def _read_json_bytes_from_tar(tar_path: str | Path, member_name: str) -> dict[str, Any]:
    with tarfile.open(tar_path, 'r') as tf:
        member = tf.getmember(member_name)
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f'Could not extract {member_name} from {tar_path}')
        return json.loads(f.read().decode('utf-8'))


def _resample(audio: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return audio
    out = []
    for ch in audio:
        y = resample_poly(ch.numpy(), target_sr, sr)
        out.append(torch.from_numpy(y).float())
    min_len = min(x.numel() for x in out)
    out = [x[:min_len] for x in out]
    return torch.stack(out, dim=0)


def _read_json_file(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_seamless_pair(row: dict[str, Any]) -> tuple[torch.Tensor, int, dict[str, Any]]:
    audio0, sr0 = sf.read(row['participant1_relpath_abs'], always_2d=True)
    audio1, sr1 = sf.read(row['participant2_relpath_abs'], always_2d=True)
    if sr0 != sr1:
        raise ValueError(f"Seamless participant sample rates differ: {sr0} vs {sr1}")

    ch0 = torch.from_numpy(audio0.T).float()[0]
    ch1 = torch.from_numpy(audio1.T).float()[0]
    n = min(ch0.numel(), ch1.numel())
    audio = torch.stack([ch0[:n], ch1[:n]], dim=0)
    meta = {
        "source_natural_stem": row.get("source_natural_stem") or row.get("id") or row.get("session_id"),
        "participant1": _read_json_file(row["participant1_json_abs"]) if row.get("participant1_json_abs") else {},
        "participant2": _read_json_file(row["participant2_json_abs"]) if row.get("participant2_json_abs") else {},
    }
    return audio, sr0, meta


def load_audio_and_meta(row: dict[str, Any], target_sr: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
    source_type = row.get('source_type', 'file')
    if row.get("participant1_relpath_abs") and row.get("participant2_relpath_abs"):
        audio, sr, meta = _load_seamless_pair(row)
    elif source_type == 'tar':
        audio, sr = _read_audio_bytes_from_tar(row['tar_path'], row['member_flac'])
        meta = _read_json_bytes_from_tar(row['tar_path'], row['member_json'])
    else:
        audio, sr = sf.read(row['audio_path'], always_2d=True)
        audio = torch.from_numpy(audio.T).float()
        with open(row['json_path'], 'r', encoding='utf-8') as f:
            meta = json.load(f)

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    if audio.shape[0] > 2:
        audio = audio[:2]

    audio = _resample(audio, sr, target_sr)
    sr = target_sr

    for seg in meta.get('redacted_segments', []) or []:
        s = int(float(seg['start_sec']) * sr)
        e = int(float(seg['end_sec']) * sr)
        s = max(s, 0)
        e = min(e, audio.shape[-1])
        if s < e:
            audio[:, s:e] = 0.0
    return audio, sr, meta
