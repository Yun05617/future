"""Utilities for training and running speech-emotion baselines."""

import os
from pathlib import Path


_BUNDLED_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if _BUNDLED_TOOLS.is_dir():
    os.environ["PATH"] = f"{_BUNDLED_TOOLS}{os.pathsep}{os.environ.get('PATH', '')}"

_PROJECT_CACHE = Path(__file__).resolve().parents[2] / ".cache"
_HUGGINGFACE_CACHE = _PROJECT_CACHE / "huggingface"
_HUGGINGFACE_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HUGGINGFACE_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HUGGINGFACE_CACHE / "hub"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_CACHE))
os.environ.setdefault("TORCH_HOME", str(_PROJECT_CACHE / "torch"))
