# Configuration reference

Everything you can configure: directories, command-line options, the ComfyUI workflow, the per-category policy, and
the training configs. For a walkthrough, see the [user guide](user-guide.md).

## Directories

All paths are relative to the project root unless you pass absolute ones. The data folders are created by you and
ignored by git.

| Path | Purpose |
|---|---|
| `original/assets/` | Extracted vanilla textures |
| `mods/assets/` | Extracted mod textures |
| `texture/` | Reference resource pack (`texture/assets/...`, optional `texture/pack.png`) |
| `datasets/lora_prep/` | Crosswalk CSVs from `prepare_lora_crosswalk.py` |
| `datasets/qwen_edit_pairs*/` | Training datasets (`control/`, `target/`, `metadata.jsonl`, caches) |
| `models/` | Training files and trained LoRAs |
| `output/` | Generated packs |
| `tools/ComfyUI/`, `tools/musubi-tuner/` | The external tools |
| `.venv/`, `.venv-comfy/`, `.venv-musubi/` | Python environments (project, ComfyUI, training) |

## `scripts/run_batch.sh`

A wrapper around `qwen_resourcepack_batch.py` with the recommended fast settings. Extra arguments are appended and
override these defaults.

| Setting | Value |
|---|---|
| `--output-root` | `output/resourcepack_1024x_pbr` |
| `--reference-root`, `--expected-reference-root` | `texture` |
| `--api-url` | `$COMFYUI_URL`, default `http://127.0.0.1:8189` |
| `--resolution` | `1024` |
| `--steps` | `4` |
| `--speed-lora` | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` |
| `--category-lora` | `block=minecraft-block-qwen-edit-2511-r16.safetensors`, `item=minecraft-item-qwen-edit-2511-r16.safetensors` |

Environment variables: `PYTHON` (interpreter, default `python3`) and `COMFYUI_URL` (ComfyUI address).

## `scripts/qwen_resourcepack_batch.py`

Generates the pack. Run `python scripts/qwen_resourcepack_batch.py --help` for the same list.

| Option | Default | Description |
|---|---|---|
| `--output-root` | required | Folder that receives the pack |
| `--source-root` | `.` | Folder that contains `original/`, `mods/`, and `texture/pack.png` |
| `--reference-root` | `texture` | Reference pack. Textures it contains are skipped, and its normal/specular maps are reused |
| `--expected-reference-root` | `texture` | Pack used by `--compare-expected` |
| `--style-reference-root` | `datasets` | Folder with the paired training datasets, used as style references for categories that use them |
| `--workflow` | `configs/qwen_gguf_lora_preflight.json` | ComfyUI workflow template |
| `--comfy-root` | `tools/ComfyUI` | ComfyUI checkout (the batch uploads inputs and reads outputs there) |
| `--api-url` | `http://127.0.0.1:8189` | ComfyUI address |
| `--resolution` | `2048` | Longest side of each output texture, in pixels |
| `--steps` | `12` | Sampler steps. Use `4` with `--speed-lora` |
| `--speed-lora` | none | Step-distillation LoRA in `models/loras/`. Runs without guidance and from full noise |
| `--workers` | `3` | Textures in flight at once, so CPU work overlaps the GPU |
| `--upscale-model` | none | Optional upscale model from `tools/ComfyUI/models/upscale_models/` applied to the input |
| `--lora-strength` | `0.8` | Strength of the category LoRA |
| `--category-lora CATEGORY=FILE` | block and item defaults | Override the LoRA for `item`, `block`, `particle`, or `environment`. Repeatable |
| `--normal-strength` | `2.0` | Strength of generated normal maps |
| `--max-textures` | `0` (all) | Stop after this many textures |
| `--texture PATH` | all | Process only this pack-relative path (for example `assets/mod/textures/block/x.png`). Repeatable |
| `--only-qwen` | off | Process only the model-driven categories (`item`, `block`, `particle`, `environment`) |
| `--only-category CATEGORY` | all | Process only one of those categories |
| `--compare-expected` | off | Compare each result with the texture of the same path under `--expected-reference-root` |
| `--min-expected-edge-similarity` | `0.0` | With `--compare-expected`, fail a texture below this edge similarity |
| `--min-expected-composite-score` | `0.0` | With `--compare-expected`, fail a texture below this composite score |
| `--overwrite` | off | Regenerate textures that already exist in the output |

Notes:

- A texture that fails is reported (`Texture failed: ...`) and counted, the run continues, and the exit code is 1 at the
  end.
- The output folder holds a lock file while a run is active. A stale lock from a dead process is removed automatically.

## Output files

For each texture `X.png` that is generated:

| File | Written when |
|---|---|
| `X.png` | Always |
| `X.png.mcmeta` | The source has animation metadata |
| `X_n.png` | Not a GUI texture. LabPBR normal: red/green = X/Y in DirectX orientation, blue = ambient occlusion (255 = none), alpha = height (255 = surface, lower = deeper) |
| `X_s.png` | Not a GUI texture. LabPBR specular: red = perceptual smoothness, green = F0 or metal id, blue = porosity, alpha = emission (255 = none) |

The pack root also gets `pack.mcmeta` (`pack_format` 34) and `pack.png` when `texture/pack.png` or `./pack.png` exists.
**How the LabPBR maps are estimated** (constants at the top of `scripts/generate_pbr_maps.py`; layout from the
[LabPBR 1.3 standard](https://shaderlabs.org/wiki/LabPBR_Material_Standard)):

| Channel | Estimated from |
|---|---|
| Normal X/Y | Gradient of the blurred luminance, scaled by `--normal-strength` |
| Normal blue (AO) | Pixels darker than their surroundings are cavities, up to 50 % occlusion |
| Normal alpha (height) | Luminance relative to the brightest point of the texture, using 60 % of the depth range; never 0 |
| Specular red (smoothness) | Brightness and local detail |
| Specular green | The file name: `iron`, `steel`, `anvil`, `chain` give metal id 230; `gold` 231; `aluminum` 232; `chrome` 233; `copper` 234; `lead` 235; `platinum` 236; `silver` 237; `metal`, `bronze`, `brass` 255 (generic metal). Everything else is a dielectric with F0 = 10 (about 4 %). Names with `ore` or `raw` are never metal, because the stone around the ore would be shaded as metal |
| Specular blue | Porosity from the name: sand 64, wool and carpet 38, planks, log, wood, and stem 12; 0 otherwise and for metals |
| Specular alpha | Names with `lava`, `fire`, `torch`, `lantern`, `light`, `magma`, `redstone`, `glow`, or `emissive` emit in proportion to brightness (up to 254); everything else is 255 (no emission) |

Reference-pack maps are reused unchanged when `--reference-root` has them. Sheets larger than the PBR size limit are
written without maps.

## Per-category policy

Defined in `scripts/purebdcraft_creative.py`.

| Category | Matched by path segment | Denoise (12-step path) | Default LoRA | Style reference |
|---|---|---|---|---|
| `block` | `block` | 0.9 | `minecraft-block-qwen-edit-2511-r16` | none |
| `item` | `item` | 0.8 | `minecraft-item-qwen-edit-2511-r16` | paired target, if available |
| `particle` | `particle` | 0.5 | workflow default | paired target, if available |
| `environment` | `environment` | 0.45 | workflow default | paired target, if available |
| other (generic) | anything else except entity and GUI | 0.65 | workflow default | paired target, if available |
| `entity`, `gui` | `entity`, `gui` | not sent to the model | none | none |

With `--speed-lora`, denoise is always 1.0 (sampling from full noise) and guidance is off. Block prompts mirror the
captions the block LoRA is trained on, and block generation deliberately uses no style reference image: a second image
made the model copy objects from it.

**Blocks with transparency** (`scripts/qwen_cutout_keying.py`):

| Constant | Value | Meaning |
|---|---|---|
| `KEY_COLOURS` | magenta, cyan, green | Candidates for the flat key colour. The one furthest from the object's colours is used |
| `KEY_DISTANCE_LOW` / `KEY_DISTANCE_HIGH` | 70 / 140 | Colour distance from the key (0 to 441) below which a pixel is background and above which it is object |
| `SILHOUETTE_GROW_PER_SOURCE_PIXEL` | 1.0 | How far, in source pixels, the drawing may extend beyond the source silhouette |
| `MIN_KEYED_COVERAGE` | 0.35 | Minimum share of the silhouette the keyed drawing must cover; below it the texture falls back to the source outline |

Only sources whose alpha is strictly 0 or 255 use this path. Partially transparent sources, animated sources, and
failed keying use the source alpha instead. Other quality constants (low-frequency colour anchoring, source-guide mix,
animation motion strength) are at the top of `scripts/qwen_texture_processing.py`.

## ComfyUI workflow (`configs/qwen_gguf_lora_preflight.json`)

The batch loads this API-format workflow and edits it for every render (image, prompt, seed, steps, LoRA, size).

| Node | Type | Setting in the file |
|---|---|---|
| 1 | `UnetLoaderGGUFAdvanced` | `qwen-image-edit-2511-Q6_K.gguf` |
| 2 | `LoraLoaderModelOnly` | `minecraft-artistic-qwen-edit-2511-r16.safetensors` (replaced per category) |
| 3 | `ModelSamplingAuraFlow` | `shift` 3.1 |
| 5 | `CLIPLoader` | `qwen_2.5_vl_7b_fp8_scaled.safetensors`, type `qwen_image`, device `default` (GPU) |
| 6 | `VAELoader` | `qwen_image_vae.safetensors` |
| 9, 10 | `TextEncodeQwenImageEditPlus` | Positive and negative conditioning |
| 14 | `KSampler` | `euler`, `simple`; steps, seed, and denoise are set per render; `cfg` 4.0 (1.0 with `--speed-lora`) |

With `--speed-lora`, the batch inserts an extra `LoraLoaderModelOnly` node for the speed LoRA and points the negative
conditioning at the positive one, because guidance is off. To use other model files, edit the names in this file.

## Dataset preparation

**`scripts/extract_jar_textures.py`**

| Option | Default | Description |
|---|---|---|
| `--jars-dir` | required | Folder with `*.jar` files |
| `--output-dir` | `mods/assets` | Destination, without the leading `assets/` segment |

**`scripts/prepare_lora_crosswalk.py`**

| Option | Default | Description |
|---|---|---|
| `--original-assets` | `original/assets` | Vanilla assets |
| `--mods-assets` | `mods/assets` | Mod assets (they override vanilla) |
| `--resource-assets` | `texture/assets` | Reference pack assets |
| `--output-dir` | `datasets/lora_prep` | Where the CSVs and `summary.json` go |

**`scripts/prepare_qwen_edit_training_dataset.py`**

| Option | Default | Description |
|---|---|---|
| `--pairs-csv` | `datasets/lora_prep/pairs.csv` | Pairs to turn into a dataset |
| `--output-root` | `datasets/qwen_edit_pairs` | New dataset folder. It must not exist yet |
| `--category` | all | `item`, `block`, `entity`, `gui`, `environment`, or `particle`. Repeatable |

## Training

**`scripts/train_qwen_edit_2511_lora.sh`**

| Option | Default | Description |
|---|---|---|
| `--dataset-config FILE` | `configs/qwen_edit_2511_dataset.toml` | musubi-tuner dataset config |
| `--output-dir DIR` | `models/lora_qwen_edit_2511_minecraft` | Where checkpoints and the final LoRA go |
| `--output-name NAME` | `minecraft-artistic-qwen-edit-2511-r16` | Base name of the LoRA file |
| `--expected-pairs N` | counted from the dataset's `metadata.jsonl` | Pair count that marks a cache as complete |

The script expects the files from `scripts/setup_models.sh --training` under `models/qwen_training/split_files/`
and the training environment at `.venv-musubi`. Fixed hyperparameters (edit the script to change them): LoRA rank
and alpha 16, learning rate 5e-5, AdamW 8-bit, two epochs, bf16 with fp8 base weights, 45 blocks swapped to CPU,
gradient checkpointing, seed 42.

**Dataset configs** (`configs/qwen_edit_2511_dataset.toml`, `..._block_dataset.toml`, `..._item_dataset.toml`) use the
musubi-tuner dataset format:

| Key | Value | Meaning |
|---|---|---|
| `resolution`, `control_resolution` | `[1024, 1024]` | Training resolution of target and control images |
| `batch_size`, `num_repeats` | 1, 1 | Batch size and repeats per epoch |
| `enable_bucket`, `bucket_no_upscale` | true, false | Aspect-ratio bucketing |
| `image_jsonl_file` | `datasets/<name>/metadata.jsonl` | Pair list written by the dataset script |
| `cache_directory` | `datasets/<name>/cache_1024` | Where latents and text-encoder outputs are cached |

See the musubi-tuner documentation (`tools/musubi-tuner/docs/dataset_config.md`) for all keys.

## Other tools

**`scripts/setup_models.sh`**: `--comfy-root DIR`, `--training`, `--training-root DIR`, `--no-lightning`.

**`scripts/generate_pbr_maps.py`**: adds normal and specular maps to an existing pack.

| Option | Default | Description |
|---|---|---|
| `--pack-root` | required | Pack to process |
| `--reference-root` | `texture` | Reference pack whose maps are reused when they exist |
| `--normal-strength` | `2.0` | Normal-map strength |
| `--overwrite` | off | Replace existing maps |

**`scripts/compare_expected_result.py`**: compares one generated texture with an expected one.

| Option | Description |
|---|---|
| `--relative-path` | Required. Pack-relative path of the texture |
| `--reference-root` | Pack holding the expected texture |
| `--output-root` | Pack holding the result (default `output/resourcepack_1024x_pbr`) |
| `--expected-path`, `--result-path` | Compare these two files directly instead |
