from __future__ import annotations

from pathlib import Path
from typing import Any

from dualturn.utils import read_csv, save_csv


MANIFEST_FIELDS = [
    "id",
    "session_id",
    "source_type",
    "audio_path",
    "json_path",
    "tar_path",
    "member_flac",
    "member_json",
    "duration_sec",
    "language",
    "session_type",
    "split",
]


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    for row in rows:
        if not row.get("id"):
            row["id"] = row.get("source_natural_stem") or row.get("session_id") or ""
        if not row.get("session_id"):
            row["session_id"] = row.get("id", "")
        if not row.get("duration_sec") and row.get("duration_s"):
            row["duration_sec"] = row["duration_s"]
        row["duration_sec"] = float(row.get("duration_sec") or 0.0)
    return rows


def save_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    save_csv(path, rows, fieldnames=MANIFEST_FIELDS)