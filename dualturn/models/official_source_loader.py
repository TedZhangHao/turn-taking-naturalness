from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_ALIAS = "_dualturn_main_src"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "dualturn-main"


def _install_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return

    mod = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    mod.__spec__ = spec
    mod.__path__ = [str(path)]
    sys.modules[name] = mod


def _ensure_alias_packages(repo_root: Path) -> None:
    base_pkg = repo_root / "dualturn"
    if not base_pkg.is_dir():
        raise FileNotFoundError(
            f"Official repo root does not contain dualturn package: {base_pkg}"
        )

    _install_namespace_package(_ALIAS, base_pkg)
    _install_namespace_package(f"{_ALIAS}.model", base_pkg / "model")
    _install_namespace_package(f"{_ALIAS}.config", base_pkg / "config")
    _install_namespace_package(f"{_ALIAS}.utils", base_pkg / "utils")


@lru_cache(maxsize=8)
def load_official_yaml(repo_root_str: str, rel_path: str) -> dict[str, Any]:
    repo_root = Path(repo_root_str)
    path = repo_root / rel_path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=8)
def get_official_turntaking_model_class(repo_root_str: str | None = None):
    repo_root = Path(repo_root_str) if repo_root_str else _default_repo_root()
    _ensure_alias_packages(repo_root)
    mod = importlib.import_module(f"{_ALIAS}.model.model")
    return mod.TurnTakingModel


def get_official_repo_root(cfg: dict[str, Any]) -> Path:
    model_cfg = cfg.get("model", {})
    explicit = model_cfg.get("official_source_repo_root")
    if explicit:
        return Path(explicit).resolve()
    return _default_repo_root().resolve()
