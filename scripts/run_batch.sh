#!/usr/bin/env bash
# Run the Qwen resource-pack batch with the recommended fast settings (Lightning 4-step LoRA, 1024 px).
# Every extra argument is passed through and overrides the defaults below, e.g.:
#   scripts/run_batch.sh --only-category item --max-textures 20 --output-root output/test
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec "${PYTHON:-python3}" -u scripts/qwen_resourcepack_batch.py \
    --output-root output/resourcepack_1024x_pbr \
    --reference-root texture \
    --expected-reference-root texture \
    --api-url "${COMFYUI_URL:-http://127.0.0.1:8189}" \
    --resolution 1024 \
    --steps 4 \
    --speed-lora Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    --category-lora block=minecraft-block-qwen-edit-2511-r16.safetensors \
    --category-lora item=minecraft-item-qwen-edit-2511-r16.safetensors \
    "$@"
