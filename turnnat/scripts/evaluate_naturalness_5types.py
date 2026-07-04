#!/usr/bin/env python3
"""Run VAP NLL scoring once and summarize five naturalness edit types."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path("dataset/turn_taking_naturalness_5types_200")
DEFAULT_OUTPUT = Path("turnnat/outputs/vap_nll_naturalness_5types_200")
DEFAULT_CHECKPOINT = Path("VAP-main/example/checkpoints/VAP_state_dict.pt")
TYPE_ORDER = [
    "early_entry",
    "shift_instead_of_hold",
    "late_response",
    "hold_instead_of_shift",
    "excessive_backchannel",
]
NLL_FIELDS = (
    "natural_mean_nll", "edited_mean_nll", "delta_mean_nll",
    "natural_tail_nll", "edited_tail_nll", "delta_tail_nll",
    "natural_dialogue_nll", "edited_dialogue_nll", "delta_nll",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def stats(values: list[float]) -> dict[str, float | int]:
    values = [value for value in values if math.isfinite(value)]
    n = len(values)
    if not values:
        return {key: float("nan") for key in ("mean", "var", "ci95_low", "ci95_high", "ci95_half_width")} | {"n": 0}
    mean = statistics.fmean(values)
    var = statistics.variance(values) if n > 1 else 0.0
    half = 1.96 * math.sqrt(var / n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "var": var,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
    }


def c_index(edited: list[float], natural: list[float]) -> dict[str, float | int]:
    edited = [value for value in edited if math.isfinite(value)]
    natural = sorted(value for value in natural if math.isfinite(value))
    concordant = tied = 0
    for value in edited:
        below = bisect.bisect_left(natural, value)
        at_or_below = bisect.bisect_right(natural, value)
        concordant += below
        tied += at_or_below - below
    total = len(edited) * len(natural)
    wrong = total - concordant - tied
    comparable = concordant + wrong
    return {
        "c_index": concordant / comparable if comparable else float("nan"),
        "c_index_concordant": concordant,
        "c_index_wrong": wrong,
        "c_index_tied": tied,
        "c_index_total_pairs": total,
        "c_index_comparable_pairs": comparable,
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid = [row for row in rows if math.isfinite(finite_float(row.get("delta_nll")))]
    natural_mean = [finite_float(row.get("original_mean_nll")) for row in valid]
    edited_mean = [finite_float(row.get("edited_mean_nll")) for row in valid]
    natural_tail = [finite_float(row.get("original_tail_nll")) for row in valid]
    edited_tail = [finite_float(row.get("edited_tail_nll")) for row in valid]
    natural_dialog = [finite_float(row.get("original_dialog_nll")) for row in valid]
    edited_dialog = [finite_float(row.get("edited_dialog_nll")) for row in valid]
    delta_mean = [edited - natural for edited, natural in zip(edited_mean, natural_mean)]
    delta_tail = [edited - natural for edited, natural in zip(edited_tail, natural_tail)]
    delta_dialog = [edited - natural for edited, natural in zip(edited_dialog, natural_dialog)]
    result: dict[str, Any] = {
        "n": len(valid),
        "natural_mean_nll": stats(natural_mean),
        "edited_mean_nll": stats(edited_mean),
        "delta_mean_nll": stats(delta_mean),
        "natural_tail_nll": stats(natural_tail),
        "edited_tail_nll": stats(edited_tail),
        "delta_tail_nll": stats(delta_tail),
        "natural_dialogue_nll": stats(natural_dialog),
        "edited_dialogue_nll": stats(edited_dialog),
        "delta_nll": stats(delta_dialog),
        "pairwise_accuracy": sum(value > 0 for value in delta_dialog) / len(delta_dialog) if delta_dialog else float("nan"),
    }
    result.update(c_index(edited_dialog, natural_dialog))
    return result


def canonicalize_types(pair_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_session = {
        str(row.get("session_id") or row.get("id") or ""): str(row.get("edit_type") or "")
        for row in manifest_rows
    }
    output = []
    for row in pair_rows:
        record = dict(row)
        session_id = str(record.get("edited_segment_id") or Path(record.get("edited_audio_path", "")).stem)
        canonical = by_session.get(session_id)
        if not canonical:
            raise KeyError(f"Pair score session is absent from manifest: {session_id}")
        record["edit_type"] = canonical
        output.append(record)
    return output


def flatten(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": label,
        "n": summary["n"],
        "pairwise_accuracy": summary["pairwise_accuracy"],
        "c_index": summary["c_index"],
        "c_index_concordant": summary["c_index_concordant"],
        "c_index_wrong": summary["c_index_wrong"],
        "c_index_tied": summary["c_index_tied"],
    }
    for field in NLL_FIELDS:
        for stat_name in ("mean", "var", "ci95_low", "ci95_high", "ci95_half_width"):
            row[f"{field}_{stat_name}"] = summary[field][stat_name]
    return row


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("Type", "N", "Mean NLL E/N", "Tail NLL E/N", "Dialog NLL E/N", "Delta NLL", "Pair Acc", "C-index")
    table = []
    for row in rows:
        table.append((
            row["type"], str(row["n"]),
            f"{row['edited_mean_nll_mean']:.4f}/{row['natural_mean_nll_mean']:.4f}",
            f"{row['edited_tail_nll_mean']:.4f}/{row['natural_tail_nll_mean']:.4f}",
            f"{row['edited_dialogue_nll_mean']:.4f}/{row['natural_dialogue_nll_mean']:.4f}",
            f"{row['delta_nll_mean']:.4f}", f"{row['pairwise_accuracy']:.4f}", f"{row['c_index']:.4f}",
        ))
    widths = [max(len(headers[i]), *(len(row[i]) for row in table)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def run_scorer(args: argparse.Namespace, manifest: Path) -> Path:
    scorer = Path(__file__).with_name("score_vap_nll_naturalness.py")
    command = [
        sys.executable, str(scorer),
        "--unnatural-manifest", str(manifest),
        "--output-dir", str(args.output_dir),
        "--checkpoint", str(args.checkpoint),
        "--device", args.device,
        "--vad-source", args.vad_source,
        "--context-s", str(args.context_s),
        "--tail-gamma", str(args.tail_gamma),
        "--lambda-mean", str(args.lambda_mean),
        "--unit-pre-s", str(args.unit_pre_s),
        "--unit-post-s", str(args.unit_post_s),
        "--min-unit-frames", str(args.min_unit_frames),
        "--no-save-frame-scores",
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("Running scorer:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)
    return args.output_dir / "artifacts/vap_nll_naturalness/pair_scores.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vad-source", choices=("silero", "rms"), default="silero")
    parser.add_argument("--context-s", type=float, default=3.0)
    parser.add_argument("--tail-gamma", type=float, default=0.25)
    parser.add_argument("--lambda-mean", type=float, default=0.5)
    parser.add_argument("--unit-pre-s", type=float, default=2.0)
    parser.add_argument("--unit-post-s", type=float, default=0.0)
    parser.add_argument("--min-unit-frames", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--pair-scores", type=Path, default=None)
    parser.add_argument("--expected-types", default=",".join(TYPE_ORDER))
    args = parser.parse_args()

    manifest = args.manifest or args.dataset_root / "manifests/test.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if args.skip_scoring:
        pair_scores = args.pair_scores or args.output_dir / "artifacts/vap_nll_naturalness/pair_scores.csv"
    else:
        pair_scores = run_scorer(args, manifest)
    if not pair_scores.is_file():
        raise FileNotFoundError(pair_scores)

    pair_rows = canonicalize_types(read_csv(pair_scores), read_csv(manifest))
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        by_type[row["edit_type"]].append(row)
    expected = [value.strip() for value in args.expected_types.split(",") if value.strip()]
    missing = [name for name in expected if name not in by_type]
    unexpected = sorted(set(by_type) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Type mismatch: missing={missing}, unexpected={unexpected}")

    summaries = {name: summarize(by_type[name]) for name in expected}
    summaries["overall"] = summarize(pair_rows)
    flat_rows = [flatten(name, summaries[name]) for name in expected] + [flatten("overall", summaries["overall"])]
    metrics_dir = args.output_dir / "artifacts/naturalness_5types_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": "vap_nll_naturalness_5types",
        "manifest": str(manifest.resolve()),
        "pair_scores": str(pair_scores.resolve()),
        "config": {
            "tail_gamma": args.tail_gamma,
            "lambda_mean": args.lambda_mean,
            "dialogue_nll_formula": "lambda_mean * MeanNLL + (1 - lambda_mean) * TailNLL",
            "delta_nll_formula": "edited_dialogue_nll - natural_dialogue_nll",
            "pairwise_accuracy_formula": "mean(delta_nll > 0)",
            "c_index_formula": "C / (C + W) over all edited-vs-natural dialogue NLL pairs; ties excluded",
        },
        "by_type": {name: summaries[name] for name in expected},
        "overall": summaries["overall"],
    }
    json_path = metrics_dir / "metrics.json"
    csv_path = metrics_dir / "metrics.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    print("\nFive-type naturalness metrics")
    print_table(flat_rows)
    print(f"\nSaved JSON -> {json_path}")
    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
