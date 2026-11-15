#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

usage() {
    printf 'Usage: %s [--dataset-config FILE] [--output-dir DIR] [--output-name NAME] [--expected-pairs N]\n' "$0"
}

dataset_config_override=""
output_dir_override=""
output_name_override=""
expected_pairs_override=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dataset-config|--output-dir|--output-name|--expected-pairs)
            [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
            case "$1" in
                --dataset-config) dataset_config_override="$2" ;;
                --output-dir) output_dir_override="$2" ;;
                --output-name) output_name_override="$2" ;;
                --expected-pairs) expected_pairs_override="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

python_bin="$project_root/.venv-musubi/bin/python"
accelerate_bin="$project_root/.venv-musubi/bin/accelerate"
musubi_root="$project_root/tools/musubi-tuner"
dataset_config="${dataset_config_override:-$project_root/configs/qwen_edit_2511_dataset.toml}"
model_root="$project_root/models/qwen_training/split_files"
dit="$model_root/diffusion_models/qwen_image_edit_2511_bf16.safetensors"
vae="$model_root/vae/qwen_image_vae.safetensors"
text_encoder="$model_root/text_encoders/qwen_2.5_vl_7b.safetensors"
output_dir="${output_dir_override:-$project_root/models/lora_qwen_edit_2511_minecraft}"
output_name="${output_name_override:-minecraft-artistic-qwen-edit-2511-r16}"
read -r cache_directory dataset_pairs < <("$python_bin" - "$dataset_config" <<'PY'
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1])
with config_path.open("rb") as source:
    dataset = tomllib.load(source)["datasets"][0]
cache_directory = Path(dataset["cache_directory"]).resolve()
metadata = Path(dataset["image_jsonl_file"])
pairs = sum(1 for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()) if metadata.is_file() else 0
print(cache_directory, pairs)
PY
)
expected_pairs="${expected_pairs_override:-$dataset_pairs}"

mkdir -p "$output_dir" "$project_root/logs/qwen_edit_2511" "$cache_directory"

latent_count="$(find "$cache_directory" -maxdepth 1 -type f -name '*_qie.safetensors' | wc -l)"
if [[ "$latent_count" -eq "$expected_pairs" ]]; then
    printf 'Phase 1/3: VAE latent cache already complete (%s/%s).\n' "$latent_count" "$expected_pairs"
else
    printf 'Phase 1/3: caching VAE latents for %s pairs.\n' "$expected_pairs"
    "$python_bin" "$musubi_root/src/musubi_tuner/qwen_image_cache_latents.py" \
        --dataset_config "$dataset_config" \
        --vae "$vae" \
        --model_version edit-2511 \
        --vae_tiling
fi

text_cache_count="$(find "$cache_directory" -maxdepth 1 -type f -name '*_qie_te.safetensors' | wc -l)"
if [[ "$text_cache_count" -eq "$expected_pairs" ]]; then
    printf 'Phase 2/3: Qwen2.5-VL cache already complete (%s/%s).\n' "$text_cache_count" "$expected_pairs"
else
    printf 'Phase 2/3: caching Qwen2.5-VL multimodal embeddings.\n'
    "$python_bin" "$musubi_root/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py" \
        --dataset_config "$dataset_config" \
        --text_encoder "$text_encoder" \
        --batch_size 1 \
        --model_version edit-2511 \
        --fp8_vl
fi

resume_args=()
latest_state="$(find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name "${output_name}*-state" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || true)"
if [[ -n "$latest_state" ]]; then
    printf 'Resuming from state: %s\n' "$latest_state"
    resume_args=(--resume "$latest_state")
fi

printf 'Phase 3/3: training the Qwen-Image-Edit-2511 LoRA for two epochs.\n'
"$accelerate_bin" launch \
    --num_cpu_threads_per_process 1 \
    --mixed_precision bf16 \
    "$musubi_root/src/musubi_tuner/qwen_image_train_network.py" \
    --dit "$dit" \
    --vae "$vae" \
    --text_encoder "$text_encoder" \
    --dataset_config "$dataset_config" \
    --model_version edit-2511 \
    --sdpa \
    --mixed_precision bf16 \
    --save_precision bf16 \
    --timestep_sampling shift \
    --weighting_scheme none \
    --discrete_flow_shift 2.2 \
    --optimizer_type adamw8bit \
    --learning_rate 5e-5 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_qwen_image \
    --network_dim 16 \
    --network_alpha 16 \
    --max_train_epochs 2 \
    --save_every_n_epochs 1 \
    --save_every_n_steps 250 \
    --save_last_n_steps 500 \
    --save_last_n_steps_state 500 \
    --save_state \
    --save_state_on_train_end \
    --seed 42 \
    --fp8_base \
    --fp8_scaled \
    --blocks_to_swap 45 \
    --output_dir "$output_dir" \
    --output_name "$output_name" \
    "${resume_args[@]}"

printf 'Training finished: %s\n' "$output_dir"