"""Loads a YAML config and fills in defaults, so every script reads settings the same way."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {
    "image_size": 320,
    "batch_size": 16,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "clahe": True,
    "augment": True,
    "pretrained": True,
    "patience": 5,
    "seed": 42,
    "models": ["densenet121", "efficientnet_b0", "resnet50", "convnext_tiny"],
    "runs_dir": "runs",
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    out = dict(DEFAULTS)
    out.update(cfg)
    for key in ("name", "labels", "manifest"):
        if key not in out:
            raise ValueError(f"Config {path} is missing '{key}'")
    out["config_path"] = str(path)
    return out


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
