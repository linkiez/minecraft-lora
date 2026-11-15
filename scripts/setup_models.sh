#!/usr/bin/env bash
# Download the model files needed to run (and optionally train) the Qwen-Image-Edit-2511 pipeline.
#
# Inference files go into the ComfyUI model folders. With --training the full bf16 files that musubi-tuner
# trains against are downloaded too (about 58 GB extra). Files that are already complete are skipped and
# interrupted downloads are resumed.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
comfy_root="$project_root/tools/ComfyUI"
training_root="$project_root/models/qwen_training/split_files"
with_training=0
with_lightning=1

usage() {
    cat <<USAGE
Usage: $0 [--comfy-root DIR] [--training] [--training-root DIR] [--no-lightning]

  --comfy-root DIR      ComfyUI checkout to install models into (default: tools/ComfyUI)
  --training            also download the bf16 files used by scripts/train_qwen_edit_2511_lora.sh
  --training-root DIR   where the training files go (default: models/qwen_training/split_files)
  --no-lightning        skip the 4-step Lightning LoRA
USAGE
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --comfy-root) comfy_root="$(cd "${2:?--comfy-root needs a value}" && pwd)"; shift 2 ;;
        --training-root) training_root="${2:?--training-root needs a value}"; shift 2 ;;
        --training) with_training=1; shift ;;
        --no-lightning) with_lightning=0; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

hf="https://huggingface.co"

# download URL DESTINATION: skip when the local size matches the server, otherwise download or resume.
download() {
    local url="$1" destination="$2" expected actual
    expected="$(curl -sIL "$url" | awk 'tolower($1)=="content-length:" {gsub("\r","",$2); size=$2} END {print size}')"
    if [[ -z "$expected" ]]; then
        echo "error: could not read the size of $url" >&2
        return 1
    fi
    # -L: measure the file a symlink points to, not the link itself.
    actual="$(stat -L -c %s "$destination" 2>/dev/null || echo 0)"
    if [[ "$actual" == "$expected" ]]; then
        echo "ok      $destination (already complete)"
        return 0
    fi
    if [[ "$actual" -gt "$expected" ]]; then
        echo "error: $destination is larger than the published file; remove it and rerun" >&2
        return 1
    fi
    mkdir -p "$(dirname "$destination")"
    echo "fetch   $destination ($((expected / 1024 / 1024)) MiB)"
    curl --fail --location --continue-at - --output "$destination" "$url"
}

models="$comfy_root/models"
# Quantised Qwen-Image-Edit-2511 (Q6_K) loaded through ComfyUI-GGUF.
download "$hf/unsloth/Qwen-Image-Edit-2511-GGUF/resolve/main/qwen-image-edit-2511-Q6_K.gguf" \
    "$models/unet/qwen-image-edit-2511-Q6_K.gguf"
# Text encoder (fp8) and VAE shared by Qwen-Image and Qwen-Image-Edit.
download "$hf/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
download "$hf/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
    "$models/vae/qwen_image_vae.safetensors"
if [[ "$with_lightning" == 1 ]]; then
    download "$hf/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors" \
        "$models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
fi

if [[ "$with_training" == 1 ]]; then
    download "$hf/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors" \
        "$training_root/diffusion_models/qwen_image_edit_2511_bf16.safetensors"
    download "$hf/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
        "$training_root/vae/qwen_image_vae.safetensors"
    download "$hf/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" \
        "$training_root/text_encoders/qwen_2.5_vl_7b.safetensors"
fi

echo "Done. Category LoRAs are not downloaded: train them (docs/user-guide.md) or supply your own."
