#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import soundfile as sf


EDIT_TYPES = (
    "early_entry",
    "late_response",
    "shift_instead_of_hold",
    "hold_instead_of_shift",
    "excessive_backchannel",
)


def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_metadata(source: dict[str, Any], *, natural_audio: str | None = None) -> dict[str, Any]:
    keep = (
        "session_id",
        "duration",
        "language",
        "session_type",
        "augmentation_type",
        "source_natural_stem",
        "turn_source",
        "vad_config",
        "edit_meta",
    )
    out = {key: source[key] for key in keep if key in source}
    if natural_audio is not None:
        out["natural_reference_wav"] = natural_audio
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Package one portable sample pair per edit type.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    with args.source_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {name: next(row for row in rows if row["edit_type"] == name) for name in EDIT_TYPES}

    manifest_rows = []
    for edit_type, row in selected.items():
        sample_dir = args.repo_root / "samples" / edit_type
        sample_dir.mkdir(parents=True, exist_ok=True)
        edited_audio = sample_dir / "edited.flac"
        natural_audio = sample_dir / "natural.flac"
        edited_json = sample_dir / "edited.json"
        natural_json = sample_dir / "natural.json"

        for source, destination in (
            (row["audio_path"], edited_audio),
            (row["natural_audio_path"], natural_audio),
        ):
            audio, sample_rate = sf.read(source, always_2d=True)
            sf.write(destination, audio, sample_rate, format="FLAC")

        edited_meta = read_json(row["json_path"])
        natural_meta = read_json(row["natural_json_path"])
        natural_rel = natural_audio.relative_to(args.repo_root).as_posix()
        write_json(edited_json, compact_metadata(edited_meta, natural_audio=natural_rel))
        write_json(natural_json, compact_metadata(natural_meta))

        manifest_rows.append({
            "session_id": row["session_id"],
            "audio_path": edited_audio.relative_to(args.repo_root).as_posix(),
            "json_path": edited_json.relative_to(args.repo_root).as_posix(),
            "natural_audio_path": natural_rel,
            "natural_json_path": natural_json.relative_to(args.repo_root).as_posix(),
            "split": "sample",
            "duration_sec": row["duration_sec"],
            "edit_type": edit_type,
        })

    output_manifest = args.repo_root / "samples" / "manifest.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} pairs to {output_manifest}")


if __name__ == "__main__":
    main()
