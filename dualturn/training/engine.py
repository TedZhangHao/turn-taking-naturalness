from __future__ import annotations

import numbers
from math import ceil
from pathlib import Path
from typing import Callable

import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dualturn.utils import move_batch_to_device


def _get_precision_policy(precision_mode: str, device: torch.device) -> dict[str, object]:
    mode = str(precision_mode).lower()

    if device.type != "cuda":
        return {
            "use_autocast": False,
            "autocast_dtype": None,
            "use_grad_scaler": False,
        }

    if mode in {"bf16", "bfloat16"}:
        return {
            "use_autocast": True,
            "autocast_dtype": torch.bfloat16,
            "use_grad_scaler": False,
        }
    if mode in {"fp16", "float16", "half"}:
        return {
            "use_autocast": True,
            "autocast_dtype": torch.float16,
            "use_grad_scaler": True,
        }
    if mode in {"fp32", "float32", "32"}:
        return {
            "use_autocast": False,
            "autocast_dtype": None,
            "use_grad_scaler": False,
        }

    raise ValueError(f"Unsupported precision mode: {precision_mode}")


def create_train_context(precision_mode: str, device: torch.device) -> tuple[dict[str, object], GradScaler]:
    policy = _get_precision_policy(precision_mode, device)
    scaler = GradScaler(device.type, enabled=bool(policy["use_grad_scaler"]))
    return policy, scaler


def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.detach().cpu())
        return None
    if isinstance(v, numbers.Number):
        return float(v)
    return None


def _infer_batch_size(batch: dict) -> int:
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return int(v.shape[0])
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, torch.Tensor):
                    return int(vv.shape[0])
        if isinstance(v, list) and v:
            return len(v)
    raise KeyError("Could not infer batch size from batch contents.")


def create_tensorboard_writer(cfg: dict, stage_name: str) -> SummaryWriter | None:
    logging_cfg = cfg.get("logging", {})
    if not bool(logging_cfg.get("use_tensorboard", True)):
        return None

    subdir = str(logging_cfg.get("tensorboard_subdir", "tensorboard"))
    log_dir = Path(cfg["paths"]["output_dir"]) / subdir / stage_name
    log_dir.mkdir(parents=True, exist_ok=True)

    flush_secs = int(logging_cfg.get("flush_secs", 30))
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
    return writer


def log_metrics_to_tensorboard(
    writer: SummaryWriter | None,
    prefix: str,
    metrics: dict,
    step: int,
) -> None:
    if writer is None:
        return

    for k, v in metrics.items():
        scalar = _to_scalar(v)
        if scalar is not None:
            writer.add_scalar(f"{prefix}/{k}", scalar, step)


def update_metric_totals(totals: dict[str, float], out: dict, batch_size: int) -> None:
    for k, v in out.items():
        scalar = _to_scalar(v)
        if scalar is not None:
            totals[k] = totals.get(k, 0.0) + scalar * batch_size


def average_metric_totals(totals: dict[str, float], n: int) -> dict[str, float]:
    return {k: v / max(n, 1) for k, v in totals.items()}


def get_eval_interval(num_batches: int, evals_per_epoch: int) -> int:
    evals = max(1, int(evals_per_epoch))
    return max(1, ceil(num_batches / evals))


def should_validate(step_in_epoch: int, num_batches: int, eval_interval: int) -> bool:
    return step_in_epoch == num_batches or (step_in_epoch % eval_interval == 0)


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    precision_mode: str,
    step_fn: Callable,
):
    model.eval()
    totals: dict[str, float] = {}
    n = 0

    policy = _get_precision_policy(precision_mode, device)

    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        with autocast(
            device_type=device.type,
            enabled=bool(policy["use_autocast"]),
            dtype=policy["autocast_dtype"],
        ):
            out = step_fn(batch)

        bs = _infer_batch_size(batch)
        update_metric_totals(totals, out, bs)
        n += bs

        if "loss" in totals:
            pbar.set_postfix(val_loss=f"{totals['loss'] / max(n, 1):.4f}")

    return average_metric_totals(totals, n)


def train_step(
    model,
    batch,
    optimizer,
    device,
    policy: dict[str, object],
    scaler: GradScaler,
    step_fn: Callable,
    grad_clip: float | None = 1.0,
):
    batch = move_batch_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)

    with autocast(
        device_type=device.type,
        enabled=bool(policy["use_autocast"]),
        dtype=policy["autocast_dtype"],
    ):
        out = step_fn(batch)
        loss = out["loss"]

    if bool(policy["use_grad_scaler"]):
        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

    return out, batch


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    precision_mode: str,
    step_fn: Callable,
    log_every: int = 20,
    grad_clip: float | None = 1.0,
    tb_writer: SummaryWriter | None = None,
    tb_prefix: str = "train_step",
    start_global_step: int = 0,
):
    model.train()

    policy, scaler = create_train_context(precision_mode, device)
    totals: dict[str, float] = {}
    n = 0

    pbar = tqdm(enumerate(loader, start=1), total=len(loader), desc="train", leave=False)
    for i, batch in pbar:
        global_step = start_global_step + i

        out, batch = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            policy=policy,
            scaler=scaler,
            step_fn=step_fn,
            grad_clip=grad_clip,
        )

        bs = batch["audio"].shape[0]
        update_metric_totals(totals, out, bs)
        n += bs

        if i % max(log_every, 1) == 0 or i == 1:
            lr = optimizer.param_groups[0]["lr"]
            step_metrics = {
                k: v for k, v in out.items() if _to_scalar(v) is not None
            }
            step_metrics["lr"] = lr
            log_metrics_to_tensorboard(tb_writer, tb_prefix, step_metrics, global_step)

            if "loss" in totals:
                pbar.set_postfix(loss=f"{totals['loss'] / max(n, 1):.4f}", lr=f"{lr:.2e}")

    return average_metric_totals(totals, n)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
