# Third-party notices

This repository contains only the project's own code, which is released under the [MIT License](LICENSE). It does
**not** include, bundle, or redistribute any of the software, models, or game assets listed below. You download them
separately, and each remains under its own license.

## Software the pipeline talks to

| Component | Used for | License | Source |
|---|---|---|---|
| ComfyUI | Runs the Qwen-Image-Edit workflow; this project only calls its HTTP API | GPL-3.0 | https://github.com/comfyanonymous/ComfyUI |
| ComfyUI-GGUF | Loads the quantised GGUF model inside ComfyUI | Apache-2.0 | https://github.com/city96/ComfyUI-GGUF |
| musubi-tuner | Trains the category LoRAs (optional) | Apache-2.0 (its README: "Other code is under the Apache License 2.0"; some bundled directories follow their upstream licenses) | https://github.com/kohya-ss/musubi-tuner |

ComfyUI is used as a separate, unmodified program over HTTP. It is not linked into or distributed with this project.

## Models

| File | Source | License (as declared on the model card) |
|---|---|---|
| `qwen-image-edit-2511-Q6_K.gguf` | https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF (quantisation of Qwen/Qwen-Image-Edit-2511) | Apache-2.0 |
| `qwen_2.5_vl_7b*.safetensors`, `qwen_image_vae.safetensors` | https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI | Apache-2.0 |
| `qwen_image_edit_2511_bf16.safetensors` (training only) | https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI | Apache-2.0 |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning | Apache-2.0 |

LoRAs that you train with this project are derived from your own training data, so their license depends on that data
(see below).

## Formats

The generated `_n` and `_s` maps follow the [LabPBR 1.3 material standard](https://shaderlabs.org/wiki/LabPBR_Material_Standard)
of the shaderLABS community. Only the published layout is implemented; no code or assets are taken from it.

## Python dependencies

Installed from PyPI by `requirements.txt`: Pillow (MIT-CMU), NumPy (BSD-3-Clause and others), and
opencv-contrib-python-headless (Apache-2.0). `requirements-dev.txt` adds pytest (MIT).

## Game and resource-pack assets

The pipeline reads textures that you supply: the vanilla game, mods, and a reference resource pack. Those assets belong
to their respective owners (Mojang Studios / Microsoft, mod authors, and the authors of the reference pack). **This
repository ships none of them**, and the MIT license of this project grants no rights to them.

- Only use assets you are allowed to use, and respect the licenses of the mods and packs you process.
- Textures generated from copyrighted source textures, and LoRAs trained on a copyrighted pack, may be derivative works.
  Check the licenses involved before you share or sell them.
- The prompts and docs mention "PureBDCraft" only to describe the visual style a user may choose to train towards.

## Trademarks and affiliation

"Minecraft" is a trademark of Mojang Synergies AB. This project is not affiliated with, endorsed by, or sponsored by
Mojang, Microsoft, BDcraft, ComfyUI, Qwen / Alibaba, or any mod author.
