"""Prepare writable, project-local caches before ML libraries are imported."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
HUGGINGFACE_CACHE = CACHE_ROOT / "huggingface"
for directory in (HUGGINGFACE_CACHE, CACHE_ROOT / "torch"):
    directory.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HUGGINGFACE_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUGGINGFACE_CACHE / "hub")
os.environ["XDG_CACHE_HOME"] = str(CACHE_ROOT)
os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
