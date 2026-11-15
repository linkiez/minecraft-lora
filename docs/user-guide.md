# User guide

This guide walks through the whole workflow: preparing assets, training LoRAs, generating a pack, using it, and fixing
common problems. Install everything first ([installation guide](installation.md)). Every option mentioned here is
listed in the [configuration guide](configuration.md).

```
JARs ──► original/ + mods/ ──┐
reference pack ──► texture/ ─┼─► crosswalk ─► dataset ─► train LoRA ─┐
                             │                                        ▼
                             └──────────────► batch generation ◄── ComfyUI + models
                                                     ▼
                                          output/<pack>/ (albedo + _n + _s)
```

## 1. Prepare your assets

The batch reads three folders from the project root. All of them are ignored by git.

| Folder | Contents | Used for |
|---|---|---|
| `original/assets/` | Textures of the vanilla game | Source textures |
| `mods/assets/` | Textures of your mods | Source textures (they override vanilla on a name clash) |
| `texture/` | The reference resource pack, unzipped (`texture/assets/...`, optionally `texture/pack.png`) | Defines which textures are skipped, supplies the pack icon, and is the training target |

**Extract the vanilla game and your mods.** Both are JAR files. Put the JAR you want in a directory and run:

```bash
# Vanilla client JAR: usually ~/.minecraft/versions/<version>/<version>.jar
mkdir -p /path/to/vanilla && cp ~/.minecraft/versions/1.21.1/1.21.1.jar /path/to/vanilla/
python scripts/extract_jar_textures.py --jars-dir /path/to/vanilla --output-dir original/assets

# Mods
python scripts/extract_jar_textures.py --jars-dir /path/to/mods --output-dir mods/assets
```

The extractor copies each `assets/<namespace>/textures/**/*.png` and its `.mcmeta` animation file, never overwrites an
existing file, and ignores paths that would escape the output folder.

**Unzip the reference pack** into `texture/` so that `texture/assets/minecraft/textures/...` exists. Use a pack whose
license lets you use it this way (see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

The textures to generate are then "everything under `original/` and `mods/` that `texture/` does not contain".

## 2. Choose where the LoRAs come from

The workflow always loads a LoRA, and none is shipped. For blocks and items the batch looks for:

- `minecraft-block-qwen-edit-2511-r16.safetensors`
- `minecraft-item-qwen-edit-2511-r16.safetensors`

in `tools/ComfyUI/models/loras/`. Either train them (next section), copy in LoRAs you already have, or point to other
files with `--category-lora block=<file>` and `--category-lora item=<file>`. Particles, environment, and other
non-structural textures use the LoRA named in the workflow file (`configs/qwen_gguf_lora_preflight.json`, node 2)
unless you override them the same way.

## 3. Train the category LoRAs

Training teaches the model your reference pack's style from pairs of "plain texture" and "the same texture in the
reference pack". Skip this section if you bring your own LoRAs.

**Step 1: pair the textures.** The crosswalk lists which vanilla/mod textures also exist in the reference pack.

```bash
python scripts/prepare_lora_crosswalk.py
```

It writes `datasets/lora_prep/pairs.csv` (the pairs) plus `base_only.csv`, `resourcepack_only.csv`, `all_base.csv`,
split files, and `summary.json`. Check `pairs_total` in the summary: the more pairs, the better the style transfer.

**Step 2: build one dataset per category.**

```bash
python scripts/prepare_qwen_edit_training_dataset.py --category block --output-root datasets/qwen_edit_pairs_block
python scripts/prepare_qwen_edit_training_dataset.py --category item  --output-root datasets/qwen_edit_pairs_item
```

Each dataset gets `control/` (the plain texture), `target/` (the reference-pack texture), a caption per pair, and
`metadata.jsonl`. Animated textures are split into frames. The command refuses to write into an existing folder, so
delete or rename an old one first. The captions use the same wording as the generation prompts in
`scripts/purebdcraft_creative.py`; if you edit one, edit the other, or the LoRA and the prompt drift apart.

**Step 3: train.** One run per category:

```bash
scripts/train_qwen_edit_2511_lora.sh \
  --dataset-config configs/qwen_edit_2511_block_dataset.toml \
  --output-dir models/lora_qwen_edit_2511_minecraft_block \
  --output-name minecraft-block-qwen-edit-2511-r16

scripts/train_qwen_edit_2511_lora.sh \
  --dataset-config configs/qwen_edit_2511_item_dataset.toml \
  --output-dir models/lora_qwen_edit_2511_minecraft_item \
  --output-name minecraft-item-qwen-edit-2511-r16
```

The script has three phases: cache VAE latents, cache text-encoder outputs, and train for two epochs. The first two
are skipped when the caches are already complete. Training saves its state periodically; if it is interrupted, rerun
the same command and it resumes from the latest state. Expect hours per category on a 24 GB card.

**Step 4: install the result.** Copy the final file into ComfyUI:

```bash
cp models/lora_qwen_edit_2511_minecraft_block/minecraft-block-qwen-edit-2511-r16.safetensors tools/ComfyUI/models/loras/
cp models/lora_qwen_edit_2511_minecraft_item/minecraft-item-qwen-edit-2511-r16.safetensors  tools/ComfyUI/models/loras/
```

The output folder also holds intermediate checkpoints that you can try instead of the final file.

## 4. Generate the resource pack

Start ComfyUI ([installation, step 5](installation.md#5-start-comfyui)), activate the project environment, then run
the batch. `scripts/run_batch.sh` applies the recommended fast settings and forwards any extra option to the batch.

```bash
source .venv/bin/activate
scripts/run_batch.sh                       # all missing textures, 1024 px, 4 steps
```

Common variations:

```bash
scripts/run_batch.sh --only-category block               # one category (block, item, particle, environment)
scripts/run_batch.sh --max-textures 10 --output-root output/test    # a quick trial in a separate folder
scripts/run_batch.sh --texture assets/mod/textures/block/gem.png --overwrite   # one texture, regenerate it
scripts/run_batch.sh --workers 4                         # more textures in flight at once
```

To call the batch directly, without the fast defaults (note that its own defaults are `--resolution 2048 --steps 12`):

```bash
python scripts/qwen_resourcepack_batch.py \
  --output-root output/resourcepack_2048x_pbr \
  --reference-root texture \
  --resolution 2048 \
  --steps 12 \
  --category-lora block=minecraft-block-qwen-edit-2511-r16.safetensors \
  --category-lora item=minecraft-item-qwen-edit-2511-r16.safetensors
```

**What happens to each texture**

| Category | Treatment |
|---|---|
| `block` | Restyled as a flat, tileable face texture. Cutouts are redrawn on a flat key colour and keyed out; animated blocks reuse one render for all frames. |
| `item`, `particle`, `environment`, other | Restyled with the category prompt, keeping silhouette and transparency. |
| `entity` | Not sent to the model: colours are quantised and the texture is upscaled so UV islands stay intact. PBR maps are generated. |
| `gui` | Same structural upscale, without PBR maps. |

**Speed and resuming.** With the Lightning LoRA one render takes about 18 seconds on an RTX 4090, and `--workers`
keeps the GPU busy while the CPU finishes other textures. A texture that already exists in the output folder is
skipped, so after an interruption you simply run the same command again; missing `_n`/`_s` maps are completed. Use
`--overwrite` to regenerate. Only one batch may write to an output folder at a time.

**Why fast mode samples from full noise.** With the 4-step Lightning LoRA the batch always samples from full noise
and uses no guidance. Starting from a partly noised copy of the source, which is what the 12-step path does,
makes the distilled model return a blurred copy of the source, so the `--speed-lora` path ignores the per-category
denoise values.

## 5. Use the pack

The output folder is a complete resource pack (`pack.mcmeta`, optional `pack.png`, and `assets/`). Zip its **contents**
so `pack.mcmeta` sits at the top level, then drop the zip into the game's `resourcepacks` folder:

```bash
cd output/resourcepack_1024x_pbr && zip -r ../my-pack.zip . -x '.qwen_resourcepack_batch.lock'
```

- `pack.mcmeta` declares `pack_format` 34 (Minecraft 1.21 to 1.21.1). Edit it for another version.
- The pack contains only the textures the reference pack lacks, so it does not overlap with the reference pack; enable
  both.
- The `_n` and `_s` files are normal and specular maps estimated from the textures (see
  [Output files](configuration.md#output-files) for their channel layout). They only matter with a shader pack that
  reads them, and they are approximations: a shader pack that expects strict LabPBR 1.3 data may interpret the
  channels differently. Without a shader pack they are ignored.
- 1024 px textures are heavy. If the game stutters, generate again at a lower `--resolution`.

## 6. Troubleshooting

**`Another qwen_resourcepack_batch run is already writing to ...` (with a pid).** A batch is still running on that
output folder. Wait for it or stop it. A lock left behind by a crashed or killed run is detected and removed
automatically, so a restart after a reboot just works.

**The batch waits forever at the start, or fails to connect.** ComfyUI is not reachable at `--api-url`. Start it
(installation, step 5) or fix the port. The batch waits up to 30 minutes for the API.

**`ComfyUI failed prompt` with a missing model or LoRA.** The filename passed to `--speed-lora` or `--category-lora`,
or named in the workflow, is not in `tools/ComfyUI/models/loras/`, or the GGUF/text-encoder/VAE files are missing.
Rerun `scripts/setup_models.sh` and check the LoRA names.

**Out of GPU memory.** Close other GPU programs and lower `--workers`. Training already uses fp8 and block swapping; on
smaller cards it may not fit.

**Blurry results.** Do not combine `--speed-lora` with a changed denoise; see the note in section 4. Without the speed
LoRA, use `--steps 12` or more.

**A texture comes out with invented details** (a medallion on a frame, an object instead of a surface). The fast model
sometimes does this on very small or abstract textures. Regenerate that texture without `--speed-lora` and with
`--steps 12`, or retrain the category LoRA with more examples.

**Cutout textures look wrong.** Cutouts that fill most of the tile, or where the model draws far less than the source
silhouette, automatically fall back to the source outline. If a specific texture still fails, regenerate it with
`--overwrite`.

**Compare against known-good textures.** To measure how close generated textures are to an expected pack, point
`--expected-reference-root` at that pack and add `--compare-expected`; each generated texture then gets an
`.expected_compare.json` next to it with edge, luminance, and colour similarity. For one texture:

```bash
python scripts/compare_expected_result.py --relative-path assets/minecraft/textures/block/stone.png \
  --reference-root texture --output-root output/resourcepack_1024x_pbr
```

## Known limitations

- Generation quality depends on the LoRAs and on how many reference-pack pairs you train on.
- Animated entity and GUI textures are upscaled, not restyled.
- Only Linux with an NVIDIA GPU has been exercised.
- Items and particles that are animated are still rendered frame by frame (with a shared seed), so subtle flicker is
  possible.
