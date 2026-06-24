from __future__ import annotations

from pathlib import Path
from typing import Any

from dualturn.utils import load_yaml


def _resolve_refs(obj: Any, root: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(v, root) for v in obj]
    if isinstance(obj, str) and "${" in obj and obj.startswith("${") and obj.endswith("}"):
        path = obj[2:-1].split(".")
        cur = root
        for p in path:
            cur = cur[p]
        return cur
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    for _ in range(5):
        cfg = _resolve_refs(cfg, cfg)
    return cfg


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    out = Path(cfg["paths"]["output_dir"])
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)