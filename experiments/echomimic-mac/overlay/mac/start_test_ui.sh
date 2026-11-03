#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export GRADIO_ANALYTICS_ENABLED=false
export GRADIO_TEMP_DIR="$PWD/output/ui_uploads"
mkdir -p "$GRADIO_TEMP_DIR"
exec .venv/bin/python -m mac.photo_webui
