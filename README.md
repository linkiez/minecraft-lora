# minecraft-lora

Restyle every texture of a Minecraft modpack into a consistent, hand-drawn look with an image-editing model, and write
the result as a ready-to-use resource pack with normal and specular maps in the LabPBR 1.3 format.

The pipeline runs [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (a quantised GGUF build)
inside [ComfyUI](https://github.com/comfyanonymous/ComfyUI), steered by LoRAs that you train on a reference resource
pack. It fills in the textures that the reference pack does not cover, for example the textures of mods the pack never
supported.

> **Assets are not included.** You supply the vanilla, mod, and reference-pack textures, and the LoRA weights. Read
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before sharing anything you generate. This project is not
> affiliated with Mojang, Microsoft, or any mod or pack author.

## What it does

- **Finds the gaps.** Collects the textures of the vanilla game and your mods, and skips every texture that the
  reference pack already provides.
- **Restyles by category.** Blocks, items, particles, and environment textures go through the model with a
  category-specific prompt and LoRA. Entity and GUI textures keep their UV layout and are upscaled structurally
  instead, because a model would break them.
- **Handles the hard cases.** Cutout blocks (plants, pots, bars) are drawn on a flat key colour and keyed out by script,
  so edges follow the drawing. Animated blocks use one render for all frames so they do not flicker. Greyscale textures
  that the game tints stay greyscale.
- **Writes a complete pack.** Output follows the original paths, includes `pack.mcmeta`, copies animation metadata,
  and estimates `_n` (normal) and `_s` (specular) maps from each texture in the [LabPBR 1.3](https://shaderlabs.org/wiki/LabPBR_Material_Standard) layout, for shader packs.
- **Is fast and resumable.** With the 4-step Lightning LoRA a texture takes about 18 seconds on an RTX 4090. Rerun the
  same command after an interruption and it continues where it stopped.

## Requirements

| | |
|---|---|
| OS | Linux (the only platform it has been run on) |
| GPU | NVIDIA with CUDA. Developed on an RTX 4090 (24 GB). Smaller cards are untested. |
| Python | 3.12 or newer for this project; musubi-tuner (training) needs 3.10 to 3.12 |
| Disk | About 30 GB for inference models, and about 60 GB more to train LoRAs |

## Quick start

```bash
git clone https://github.com/linkiez/minecraft-lora.git
cd minecraft-lora
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests            # sanity check

# One-time: install ComfyUI and ComfyUI-GGUF, then download the models.
# See docs/installation.md for the exact steps.
scripts/setup_models.sh

# Start ComfyUI in another terminal, then smoke-test three textures:
source .venv/bin/activate
scripts/run_batch.sh --max-textures 3 --output-root output/smoke
```

The batch needs your textures in `original/`, `mods/`, and `texture/`, and one LoRA per category. The
[user guide](docs/user-guide.md) explains how to prepare them and how to train the LoRAs.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/installation.md) | ComfyUI, models, optional training tools, verification |
| [User guide](docs/user-guide.md) | Preparing assets, training LoRAs, generating a pack, using and checking the result, troubleshooting |
| [Configuration](docs/configuration.md) | Every command-line option, default, directory, and the per-category policy |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Licenses of the tools, models, and assets involved |
| [Contributing](CONTRIBUTING.md) | Running the tests and sending changes |

## Repository layout

```
scripts/          command-line tools (batch generation, dataset preparation, training, PBR maps)
configs/          ComfyUI workflow and musubi-tuner dataset configs
tests/unit/       automated tests
docs/             guides
```

These folders are created by you and are ignored by git: `original/`, `mods/`, `texture/`, `datasets/`, `models/`,
`output/`, `logs/`, and `tools/`.

## License

[MIT](LICENSE). The tools, models, and game assets the pipeline uses are covered by their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
