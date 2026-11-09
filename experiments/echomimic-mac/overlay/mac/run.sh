#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PATH="/opt/homebrew/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python infer_audio2vid.py --device mps --config configs/prompts/animation_mac.yaml -W 256 -H 256 -L 24 --fps 12 --context_frames 12 --context_overlap 3 --steps 10 "$@"
