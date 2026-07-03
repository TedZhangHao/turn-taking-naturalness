#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from tqdm import tqdm

from train_fvad_head import IGNORE_INDEX, fvad_targets

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


class NaturalnessFiveTypeEvaluator:
    def __init__(
        self,
        *,
        manifest: Path,
        output_dir: Path,
        sample_rate: int,
        samples_per_frame: int,
        model_frame_hz: float,
        bin_frames_50hz: list[int],
        threshold_ratio: float,
        bernoulli_head_reduction: str,
        batch_size: int,
        limit: int | None,
        vad_source: str,
        rms_threshold: float,
        silero_threshold: float,
        silero_min_speech_ms: int,
        silero_min_silence_ms: int,
        clean_min_speech_ms: int,
        clean_min_silence_ms: int,
        context_s: float,
        tail_gamma: float,
        lambda_mean: float,
        unit_pre_s: float,
        unit_post_s: float,
        min_unit_frames: int,
        min_utterance_s: float,
        utterance_merge_gap_s: float,
        utterance_merge_other_max_ratio: float,
        unit_mode: str,
    ) -> None:
        from dualturn.scripts import evaluate_naturalness_5types as nat5
        from dualturn.scripts import score_vap_nll_naturalness as vap_nll_base

        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        self.base = vap_nll_base
        self.nat5 = nat5
        self.manifest = manifest
        self.output_dir = output_dir
        self.sample_rate = int(sample_rate)
        self.samples_per_frame = int(samples_per_frame)
        self.model_frame_hz = float(model_frame_hz)
        self.bin_frames_50hz = list(bin_frames_50hz)
        self.threshold_ratio = float(threshold_ratio)
        self.bernoulli_head_reduction = str(bernoulli_head_reduction)
        if self.bernoulli_head_reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid Bernoulli head reduction: {self.bernoulli_head_reduction}")
        self.batch_size = int(batch_size)
        self.vad_source = str(vad_source)
        self.rms_threshold = float(rms_threshold)
        self.silero_threshold = float(silero_threshold)
        self.silero_min_speech_ms = int(silero_min_speech_ms)
        self.silero_min_silence_ms = int(silero_min_silence_ms)
        self.clean_min_speech_ms = int(clean_min_speech_ms)
        self.clean_min_silence_ms = int(clean_min_silence_ms)
        self.context_s = float(context_s)
        self.tail_gamma = float(tail_gamma)
        self.lambda_mean = float(lambda_mean)
        self.unit_pre_s = float(unit_pre_s)
        self.unit_post_s = float(unit_post_s)
        self.min_unit_frames = int(min_unit_frames)
        self.min_utterance_s = float(min_utterance_s)
        self.utterance_merge_gap_s = float(utterance_merge_gap_s)
        self.utterance_merge_other_max_ratio = float(utterance_merge_other_max_ratio)
        self.unit_mode = str(unit_mode)
        self.rows = self.base.read_manifest(manifest)
        self.limit = limit
        if limit is not None:
            self.rows = self.rows[: int(limit)]
        if self.batch_size < 1:
            raise ValueError("naturalness batch_size must be >= 1")
        self.silero_models = self.base.load_silero_models() if self.vad_source == "silero" else None
        print(f"Naturalness 5types eval: {len(self.rows)} pairs from {manifest}; pair_batch_size={self.batch_size}")

    def _pair_segments(self, row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        session_id = str(row.get("session_id") or row.get("id") or Path(row["audio_path"]).stem)
        pair_id = str(row.get("pair_id") or session_id)
        edit_type = str(row.get("edit_type") or "")
        natural_audio = Path(row.get("natural_audio_path") or "")
        if not natural_audio:
            meta = self.base.read_json(row["json_path"])
            natural_audio = self.base.natural_reference_from_meta(meta, row["json_path"])
        edited = {
            "segment_id": session_id,
            "condition": "edited",
            "version": "edited",
            "pair_id": pair_id,
            "edit_type": edit_type,
            "audio_path": Path(row["audio_path"]),
        }
        natural = {
            "segment_id": f"{pair_id}__natural_ref_for__{session_id}",
            "condition": "natural",
            "version": "original",
            "pair_id": pair_id,
            "edit_type": edit_type,
            "audio_path": natural_audio,
        }
        return natural, edited

    def _prepare_segment(self, item: dict[str, Any]) -> dict[str, Any]:
        model_audio = self.base.load_stereo_audio(Path(item["audio_path"]), self.sample_rate)
        vad_audio = self.base.load_stereo_audio(Path(item["audio_path"]), 16_000)
        duration_s = float(model_audio.shape[-1] / self.sample_rate)
        vad_pack = self.base.observed_vad_50hz(
            vad_audio,
            sr=16_000,
            vad_source=self.vad_source,
            silero_models=self.silero_models,
            rms_threshold=self.rms_threshold,
            silero_threshold=self.silero_threshold,
            silero_min_speech_ms=self.silero_min_speech_ms,
            silero_min_silence_ms=self.silero_min_silence_ms,
            clean_min_speech_ms=self.clean_min_speech_ms,
            clean_min_silence_ms=self.clean_min_silence_ms,
        )
        units = self.base.extract_utterance_units(
            vad_pack["clean"],
            frame_hz=50.0,
            duration_s=duration_s,
            unit_pre_s=self.unit_pre_s,
            unit_post_s=self.unit_post_s,
            min_utterance_s=self.min_utterance_s,
            unit_mode=self.unit_mode,
            utterance_merge_gap_s=self.utterance_merge_gap_s,
            utterance_merge_other_max_ratio=self.utterance_merge_other_max_ratio,
        )
        return {**item, "audio": model_audio, "duration_s": duration_s, "vad_clean": vad_pack["clean"], "units": units}

    def _vad_to_model_frames(self, vad_50hz: np.ndarray, target_frames: int) -> np.ndarray:
        out = np.zeros((target_frames, 2), dtype=np.int8)
        if target_frames <= 0:
            return out
        for i in range(target_frames):
            lo = int(math.floor(i * 50.0 / self.model_frame_hz))
            hi = int(math.floor((i + 1) * 50.0 / self.model_frame_hz))
            hi = max(hi, lo + 1)
            if lo < vad_50hz.shape[0]:
                out[i] = (vad_50hz[lo:min(hi, vad_50hz.shape[0])].mean(axis=0) >= 0.5).astype(np.int8)
        return out

    @torch.no_grad()
    def _score_prepared_batch(
        self,
        model: nn.Module,
        prepared: list[dict[str, Any]],
        *,
        device: torch.device,
        use_autocast: bool,
        autocast_dtype: torch.dtype | None,
    ) -> list[dict[str, Any]]:
        max_samples = max(int(item["audio"].shape[-1]) for item in prepared)
        audio_batch = torch.stack([
            F.pad(item["audio"], (0, max_samples - int(item["audio"].shape[-1])))
            for item in prepared
        ]).to(device)
        with autocast(device_type=device.type, enabled=use_autocast, dtype=autocast_dtype):
            logits_batch = model(audio_batch)
        logits_batch = logits_batch.float()
        results = []
        for index, item in enumerate(prepared):
            logits = logits_batch[index:index + 1]
            T = int(logits.shape[1])
            valid_frames = min(T, int(math.ceil(float(item["audio"].shape[-1]) / self.samples_per_frame)))
            frame_valid_mask = torch.zeros((1, T), dtype=torch.float32, device=device)
            frame_valid_mask[:, :valid_frames] = 1.0
            vad_50 = torch.from_numpy(item["vad_clean"].T.astype(np.float32)).unsqueeze(0).to(device)
            states, valid = fvad_targets(
                vad_50,
                frame_valid_mask,
                self.bin_frames_50hz,
                model_frame_hz=self.model_frame_hz,
                threshold_ratio=self.threshold_ratio,
                target_scheme=model.target_scheme,
            )
            n = min(T, int(states.shape[1]))
            nll = np.full((n,), np.nan, dtype=np.float32)
            targets = np.full((n,), IGNORE_INDEX, dtype=np.int64)
            if n > 0:
                states_n = states[:, :n]
                valid_n = valid[:, :n]
                if bool(valid_n.any()):
                    powers = torch.tensor([1 << i for i in range(8)], dtype=torch.long, device=device)
                    packed_targets = ((states_n >= self.threshold_ratio).long() * powers).sum(dim=-1)
                    if model.objective_type == "categorical256":
                        losses = F.cross_entropy(
                            logits[:, :n].reshape(-1, logits.shape[-1]),
                            packed_targets.reshape(-1),
                            reduction="none",
                        ).reshape(1, n)
                    elif model.objective_type == "bernoulli8":
                        per_bit_losses = F.binary_cross_entropy_with_logits(
                            logits[:, :n], states_n, reduction="none"
                        )
                        if self.bernoulli_head_reduction == "sum":
                            losses = per_bit_losses.sum(dim=-1)
                        else:
                            losses = per_bit_losses.mean(dim=-1)
                    else:
                        raise ValueError(f"Unsupported objective_type={model.objective_type}")
                    valid_np = valid_n.squeeze(0).detach().cpu().numpy().astype(bool)
                    nll[valid_np] = losses.squeeze(0).detach().cpu().numpy()[valid_np]
                    targets[valid_np] = packed_targets.squeeze(0).detach().cpu().numpy()[valid_np]
            vad_model = self._vad_to_model_frames(item["vad_clean"], len(nll))
            unit_rows = self.base.unit_rows_from_units(
                nll,
                item["units"],
                vad=vad_model,
                segment_id=item["segment_id"],
                frame_hz=self.model_frame_hz,
                context_s=self.context_s,
                min_unit_frames=self.min_unit_frames,
            )
            aggregate = self.base.aggregate_unit_rows(unit_rows, gamma=self.tail_gamma, lam=self.lambda_mean)
            aggregate.update({
                "segment_id": item["segment_id"],
                "condition": item["condition"],
                "version": item["version"],
                "pair_id": item["pair_id"],
                "edit_type": item["edit_type"],
                "audio_path": str(item["audio_path"]),
                "duration_s": item["duration_s"],
                "num_nll_frames": int(np.isfinite(nll).sum()),
                "units": unit_rows,
                "targets": targets,
            })
            results.append(aggregate)
        del audio_batch, logits_batch
        return results

    def _summary_payload(self, flat_rows: list[dict[str, Any]]) -> dict[str, float | int]:
        payload: dict[str, float | int] = {}
        wanted = {
            "natural_mean_nll_mean": "natural_mean_nll",
            "edited_mean_nll_mean": "edited_mean_nll",
            "delta_mean_nll_mean": "delta_mean_nll",
            "natural_tail_nll_mean": "natural_tail_nll",
            "edited_tail_nll_mean": "edited_tail_nll",
            "delta_tail_nll_mean": "delta_tail_nll",
            "natural_dialogue_nll_mean": "natural_dialogue_nll",
            "edited_dialogue_nll_mean": "edited_dialogue_nll",
            "delta_nll_mean": "delta_nll",
            "pairwise_accuracy": "pairwise_acc",
            "c_index": "c_index",
            "n": "n",
        }
        for row in flat_rows:
            label = str(row["type"])
            prefix = f"naturalness/{label}"
            for src, dst in wanted.items():
                value = row.get(src)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    payload[f"{prefix}/{dst}"] = float(value) if src != "n" else int(value)
        return payload

    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        epoch: int | None,
        global_step: int,
        use_autocast: bool,
        autocast_dtype: torch.dtype | None,
    ) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        pair_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        unit_rows: list[dict[str, Any]] = []
        for batch_start in tqdm(range(0, len(self.rows), self.batch_size), desc=f"naturalness step{global_step}", leave=False):
            batch_rows = self.rows[batch_start:batch_start + self.batch_size]
            infos = [self._pair_segments(row) for row in batch_rows]
            prepared = []
            for natural, edited in infos:
                prepared.append(self._prepare_segment(natural))
                prepared.append(self._prepare_segment(edited))
            scored = self._score_prepared_batch(
                model,
                prepared,
                device=device,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
            )
            for local_index, (natural_info, edited_info) in enumerate(infos):
                natural = scored[2 * local_index]
                edited = scored[2 * local_index + 1]
                for segment in (natural, edited):
                    segment_rows.append({
                        "segment_id": segment["segment_id"],
                        "condition": segment["condition"],
                        "pair_id": segment["pair_id"],
                        "version": segment["version"],
                        "edit_type": segment["edit_type"],
                        "audio_path": segment["audio_path"],
                        "duration_s": segment["duration_s"],
                        "mean_nll": segment["mean_nll"],
                        "tail_nll": segment["tail_nll"],
                        "dialog_nll": segment["dialog_nll"],
                        "nat_score": segment["nat_score"],
                        "num_units": segment["num_units"],
                        "tail_k": segment["tail_k"],
                        "num_nll_frames": segment["num_nll_frames"],
                    })
                    unit_rows.extend(segment["units"])
                delta_nll = float(edited["dialog_nll"]) - float(natural["dialog_nll"])
                pair_rows.append({
                    "pair_id": natural["pair_id"],
                    "edit_type": natural["edit_type"],
                    "original_segment_id": natural["segment_id"],
                    "edited_segment_id": edited["segment_id"],
                    "original_audio_path": natural["audio_path"],
                    "edited_audio_path": edited["audio_path"],
                    "original_mean_nll": natural["mean_nll"],
                    "original_tail_nll": natural["tail_nll"],
                    "original_dialog_nll": natural["dialog_nll"],
                    "edited_mean_nll": edited["mean_nll"],
                    "edited_tail_nll": edited["tail_nll"],
                    "edited_dialog_nll": edited["dialog_nll"],
                    "delta_nll": delta_nll,
                    "edited_more_unnatural": bool(delta_nll > 0),
                    "original_num_units": natural["num_units"],
                    "edited_num_units": edited["num_units"],
                })
            del prepared, scored
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in pair_rows:
            by_type.setdefault(str(row["edit_type"]), []).append(row)
        expected_all = list(self.nat5.TYPE_ORDER)
        expected = expected_all if self.limit is None else [name for name in expected_all if name in by_type]
        missing = [name for name in expected if name not in by_type]
        unexpected = sorted(set(by_type) - set(expected_all))
        if missing or unexpected:
            raise ValueError(f"Naturalness type mismatch: missing={missing}, unexpected={unexpected}")
        summaries = {name: self.nat5.summarize(by_type[name]) for name in expected}
        summaries["overall"] = self.nat5.summarize(pair_rows)
        flat_rows = [self.nat5.flatten(name, summaries[name]) for name in expected]
        flat_rows.append(self.nat5.flatten("overall", summaries["overall"]))
        out_dir = self.output_dir / f"step_{global_step:08d}"
        _write_csv(out_dir / "pair_scores.csv", pair_rows)
        _write_csv(out_dir / "segment_scores.csv", segment_rows)
        _write_csv(out_dir / "units.csv", unit_rows)
        _write_csv(out_dir / "metrics.csv", flat_rows)
        payload = {
            "metric": "future_vad_naturalness_5types",
            "objective_type": model.objective_type,
            "fvad_target_scheme": model.target_scheme,
            "frame_nll_reduction": (
                "categorical_ce" if model.objective_type == "categorical256"
                else f"bernoulli_{self.bernoulli_head_reduction}"
            ),
            "epoch": epoch,
            "global_step": global_step,
            "manifest": str(self.manifest),
            "model_frame_hz": self.model_frame_hz,
            "label_frame_hz": 50.0,
            "num_pairs": len(pair_rows),
            "batch_size": self.batch_size,
            "by_type": {name: summaries[name] for name in expected},
            "overall": summaries["overall"],
        }
        (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if was_training:
            model.train(True)
        return {"summary": payload, "flat_rows": flat_rows, "metrics": self._summary_payload(flat_rows), "output_dir": str(out_dir)}
