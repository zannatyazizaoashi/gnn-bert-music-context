from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path = "config.yaml", profile: str | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if profile:
        if profile not in config["profiles"]:
            raise ValueError(f"Unknown profile {profile!r}; choose {list(config['profiles'])}")
        overrides = config["profiles"][profile]
        config["active_profile"] = profile
        for key in ("epochs", "batch_size"):
            if key in overrides:
                config["training"][key] = overrides[key]
        config["data"]["max_tracks"] = overrides.get("max_tracks")
    return config


def resolve_paths(config: dict[str, Any], root: str | Path = ".") -> dict[str, Path]:
    root = Path(root).resolve()
    return {key: (root / value).resolve() for key, value in config["paths"].items()}


def ensure_project_dirs(config: dict[str, Any], root: str | Path = ".") -> dict[str, Path]:
    paths = resolve_paths(config, root)
    for key, path in paths.items():
        if key not in {"manifest", "label_map", "metadata_dir", "audio_dir"}:
            path.mkdir(parents=True, exist_ok=True)
    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = (p for p in model.parameters() if p.requires_grad) if trainable_only else model.parameters()
    return sum(p.numel() for p in params)

