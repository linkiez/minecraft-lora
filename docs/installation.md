# Installation

This guide sets up everything needed to generate a resource pack. Training your own LoRAs needs a few extra pieces,
marked **optional**. Versions below are the ones the project was developed and tested with.

## 1. Prerequisites

- Linux, `git`, `curl`, and `python3` 3.12 or newer.
- An NVIDIA GPU with a recent driver and CUDA. Developed on an RTX 4090 (24 GB VRAM) with 91 GB of RAM.
- Free disk space: about 30 GB for the inference models. Training adds about 60 GB for the full-precision files plus
  the dataset caches.

## 2. This project

```bash
git clone https://github.com/linkiez/minecraft-lora.git
cd minecraft-lora
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests
```

The batch script itself only needs Pillow, NumPy, and OpenCV: it talks to ComfyUI over HTTP, so this one environment
is enough to run it. ComfyUI and the training tools use their own environments (below), because they need PyTorch.

## 3. ComfyUI

Clone ComfyUI into `tools/ComfyUI` (the default location the scripts expect) and create its own environment.
Tested with ComfyUI v0.28.0 (commit `71b73e3b`).

```bash
mkdir -p tools
git clone https://github.com/comfyanonymous/ComfyUI.git tools/ComfyUI
python3.12 -m venv .venv-comfy
source .venv-comfy/bin/activate
```

Install PyTorch for your CUDA version first, using the command from <https://pytorch.org/get-started/locally/>
(the tested environment used torch 2.13 with CUDA 13.0), then ComfyUI's dependencies:

```bash
pip install -r tools/ComfyUI/requirements.txt
```

Install the GGUF loader node (tested with commit `6ea2651`):

```bash
git clone https://github.com/city96/ComfyUI-GGUF.git tools/ComfyUI/custom_nodes/ComfyUI-GGUF
pip install -r tools/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
```

## 4. Models

`scripts/setup_models.sh` downloads the inference models into the ComfyUI folders. It skips files that are already
complete and resumes interrupted downloads, so it is safe to rerun.

```bash
scripts/setup_models.sh
```

| File | Goes to | Size | Source |
|---|---|---|---|
| `qwen-image-edit-2511-Q6_K.gguf` | `tools/ComfyUI/models/unet/` | 17.9 GB | `unsloth/Qwen-Image-Edit-2511-GGUF` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `tools/ComfyUI/models/text_encoders/` | 9.4 GB | `Comfy-Org/Qwen-Image_ComfyUI` |
| `qwen_image_vae.safetensors` | `tools/ComfyUI/models/vae/` | 0.25 GB | `Comfy-Org/Qwen-Image_ComfyUI` |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `tools/ComfyUI/models/loras/` | 0.85 GB | `lightx2v/Qwen-Image-Edit-2511-Lightning` |

Useful options: `--comfy-root DIR` installs into another ComfyUI checkout, and `--no-lightning` skips the speed LoRA
(you then run the slower 12-step path; see the [configuration guide](configuration.md)).

**No LoRA for your art style is downloaded.** The workflow needs a category LoRA for blocks and items, and you must
supply or train one. See [Training LoRAs](user-guide.md#3-train-the-category-loras).

## 5. Start ComfyUI

```bash
cd tools/ComfyUI
../../.venv-comfy/bin/python main.py --port 8189 --listen 127.0.0.1
```

Keep it listening on `127.0.0.1`: ComfyUI has no authentication. Check that it answers:

```bash
curl -s http://127.0.0.1:8189/system_stats | head -c 200
```

Use `--api-url` (or the `COMFYUI_URL` variable of `scripts/run_batch.sh`) if you use another port.

## 6. Training tools (optional)

Needed only to train LoRAs with [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). It requires Python 3.10 to
3.12. Tested with commit `8934cfb`.

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git tools/musubi-tuner
python3.12 -m venv .venv-musubi
source .venv-musubi/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # pick your CUDA build
pip install -e tools/musubi-tuner
accelerate config
```

Then download the full-precision training files (about 58 GB):

```bash
scripts/setup_models.sh --training
```

This places `qwen_image_edit_2511_bf16.safetensors`, the VAE, and the unquantised text encoder under
`models/qwen_training/split_files/`, where `scripts/train_qwen_edit_2511_lora.sh` expects them.

## 7. Verify the installation

With ComfyUI running and your assets in place (see the [user guide](user-guide.md#1-prepare-your-assets)):

```bash
source .venv/bin/activate
scripts/run_batch.sh --max-textures 3 --output-root output/smoke
```

Three textures should appear under `output/smoke/assets/`, each with `_n` and `_s` maps next to it (GUI textures get
no maps). If something fails, see [Troubleshooting](user-guide.md#6-troubleshooting).
