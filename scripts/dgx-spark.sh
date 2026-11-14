#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Avoid per-shape cuDNN planning latency. An explicit caller setting takes priority.
export IRODORI_SDPA_CUDNN="${IRODORI_SDPA_CUDNN:-0}"
if [ "$#" -eq 0 ]; then
    set -- gradio_app_voicedesign.py
fi
exec uv run --locked --extra dgx-spark python "$@"
