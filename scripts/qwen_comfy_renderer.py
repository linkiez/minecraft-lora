"""ComfyUI workflow construction and rendering client for the Qwen resource pack batch."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

from purebdcraft_creative import PureBDCraftCreativeStrategy, deterministic_seed

UPSCALE_SAVE_NODE_ID = "16"
SPEED_LORA_CFG = 1.0  # distilled models run without classifier-free guidance
DEFAULT_LATENT_SIZE = (1024, 1024)
LATENT_TARGET_PIXELS = 1024 * 1024
LATENT_SIZE_MULTIPLE = 16
STYLE_TRANSFER_PROMPT_PREFIX = (
    "Image 1 is the source texture and defines identity, geometry, layout, orientation, and transparency. "
    "Preserve the unique palette, markings, symbols, and details from Image 1 so the mod texture remains identifiable. "
    "Image 2 is the PureBDCraft style reference; transfer only its bold outlined cartoon language, "
    "clean color regions, controlled cel shading, material rendering, and edge treatment. "
    "Do not copy the object, layout, symbols, or colors from Image 2. "
)


def latent_size_for(frame_size: tuple[int, int]) -> tuple[int, int]:
    """Return a latent size with ~1MP area that keeps the aspect ratio of ``frame_size``."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        return DEFAULT_LATENT_SIZE
    scale = (LATENT_TARGET_PIXELS / (width * height)) ** 0.5

    def snap(value: float) -> int:
        return max(LATENT_SIZE_MULTIPLE, round(value * scale / LATENT_SIZE_MULTIPLE) * LATENT_SIZE_MULTIPLE)

    return snap(width), snap(height)


def build_workflow(
    template: dict,
    input_name: str,
    prompt: str,
    output_prefix: str,
    seed: int,
    steps: int,
    lora_strength: float,
    denoise: float = 1.0,
    lora_name: str | None = None,
    upscale_model_name: str | None = None,
    style_reference_name: str | None = None,
    latent_size: tuple[int, int] = DEFAULT_LATENT_SIZE,
    negative_prompt: str = "",
    speed_lora_name: str | None = None,
    cfg: float | None = None,
) -> dict:
    workflow = copy.deepcopy(template)
    nodes = workflow["prompt"]
    nodes["2"]["inputs"]["strength_model"] = lora_strength
    if lora_name is not None:
        nodes["2"]["inputs"]["lora_name"] = lora_name
    if speed_lora_name:
        # Step-distillation LoRA chained after the style LoRA, before the sampling-shift node.
        nodes["22"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["2", 0], "lora_name": speed_lora_name, "strength_model": 1.0},
        }
        nodes["3"]["inputs"]["model"] = ["22", 0]
    if cfg is not None:
        nodes["14"]["inputs"]["cfg"] = cfg
        if cfg <= 1.0:
            # Without guidance the negative conditioning is never used; pointing it at the positive one stops
            # ComfyUI from running a second (CPU-bound) text+image encode per texture.
            nodes["14"]["inputs"]["negative"] = nodes["14"]["inputs"]["positive"]
    nodes["7"]["inputs"]["image"] = input_name
    nodes["9"]["inputs"]["prompt"] = prompt
    if negative_prompt:
        nodes["10"]["inputs"]["prompt"] = negative_prompt
    nodes["14"]["inputs"]["seed"] = seed
    nodes["14"]["inputs"]["steps"] = steps
    nodes["14"]["inputs"]["denoise"] = denoise
    nodes[UPSCALE_SAVE_NODE_ID]["inputs"]["filename_prefix"] = output_prefix
    if upscale_model_name:
        nodes["17"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale_model_name},
        }
        nodes["18"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["17", 0],
                "image": ["7", 0],
            },
        }
        nodes["8"]["inputs"]["image"] = ["18", 0]
    if style_reference_name:
        nodes["19"] = {
            "class_type": "LoadImage",
            "inputs": {"image": style_reference_name},
        }
        nodes["20"] = {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["19", 0]},
        }
        nodes["9"]["inputs"]["image2"] = ["20", 0]
        nodes["10"]["inputs"]["image2"] = ["20", 0]
        nodes["21"] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": latent_size[0], "height": latent_size[1], "batch_size": 1},
        }
        nodes["14"]["inputs"]["latent_image"] = ["21", 0]
        # An empty latent carries no source structure, so it must be fully denoised; the source
        # geometry is conveyed through the Qwen image conditioning instead.
        nodes["14"]["inputs"]["denoise"] = 1.0
    return workflow


def list_installed_upscale_models(comfy_root: Path) -> list[str]:
    models_root = comfy_root / "models" / "upscale_models"
    if not models_root.is_dir():
        return []
    return sorted(
        path.name
        for path in models_root.iterdir()
        if path.is_file() and not path.name.startswith("put_")
    )


def resolve_upscale_model_name(comfy_root: Path, requested_name: str) -> str | None:
    available = list_installed_upscale_models(comfy_root)
    if requested_name:
        if requested_name not in available:
            available_text = ", ".join(available) if available else "none"
            raise FileNotFoundError(
                f"ComfyUI upscale model {requested_name!r} was not found in {comfy_root / 'models' / 'upscale_models'} "
                f"(available: {available_text})"
            )
        return requested_name
    return None


class LazyRenderer:
    def __init__(self, renderer: ComfyRenderer) -> None:
        self.renderer = renderer
        self.ready = False

    def __call__(
        self,
        frame: Image.Image,
        prompt: str,
        frame_key: str,
        seed_key: str | None = None,
    ) -> Image.Image:
        if not self.ready:
            self.renderer.wait_until_ready()
            self.ready = True
        return self.renderer(frame, prompt, frame_key, seed_key)


class ComfyRenderer:
    def __init__(
        self,
        api_url: str,
        comfy_root: Path,
        workflow_template: dict,
        steps: int,
        lora_strength: float,
        category_loras: dict[str, str],
        upscale_model_name: str | None = None,
        style_reference_root: Path | None = None,
        speed_lora_name: str | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.input_root = comfy_root / "input" / "resourcepack_batch"
        self.output_root = comfy_root / "output"
        self.workflow_template = workflow_template
        self.steps = steps
        self.lora_strength = lora_strength
        self.upscale_model_name = upscale_model_name
        self.speed_lora_name = speed_lora_name
        self.creative_strategy = PureBDCraftCreativeStrategy(style_reference_root, category_loras)
        self.input_root.mkdir(parents=True, exist_ok=True)

    def wait_until_ready(self, timeout: int = 1_800) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.request_json("/system_stats")
                return
            except (OSError, json.JSONDecodeError):
                time.sleep(5)
        raise TimeoutError(f"ComfyUI API was not ready after {timeout} seconds")

    def request_json(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    def wait_for_history(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + 1_800
        while time.monotonic() < deadline:
            history = self.request_json(f"/history/{prompt_id}")
            if prompt_id in history:
                result = history[prompt_id]
                if result.get("status", {}).get("status_str") != "success":
                    raise RuntimeError(f"ComfyUI failed prompt {prompt_id}: {result.get('status')}")
                return result
            time.sleep(2)
        raise TimeoutError(f"ComfyUI timed out prompt {prompt_id} after 1800 seconds")

    def __call__(
        self,
        frame: Image.Image,
        prompt: str,
        frame_key: str,
        seed_key: str | None = None,
    ) -> Image.Image:
        plan = self.creative_strategy.plan_for(Path(frame_key), frame, frame_key)
        if plan is None:
            raise ValueError(f"Texture is not eligible for creative generation: {frame_key}")
        digest = hashlib.sha256(frame_key.encode("utf-8")).hexdigest()[:20]
        input_name = f"resourcepack_batch/{digest}.png"
        input_path = self.input_root / f"{digest}.png"
        style_input_name = f"resourcepack_batch/{digest}_style.png"
        style_input_path = self.input_root / f"{digest}_style.png"
        # Unique per submission: ComfyUI caches identical prompts and would otherwise report an output
        # file from an earlier run that has already been cleaned up.
        output_prefix = f"resourcepack_batch/{digest}_{uuid.uuid4().hex[:8]}"
        output_path: Path | None = None
        # The caller's prompt is the full instruction for this frame (e.g. with a background clause); the plan's
        # prompt is only the default.
        prompt = prompt or plan.prompt
        style_reference = plan.style_reference
        if style_reference is not None:
            prompt = f"{STYLE_TRANSFER_PROMPT_PREFIX}{prompt}"
        try:
            frame.save(input_path)
            if style_reference is not None:
                shutil.copy2(style_reference, style_input_path)
            workflow = build_workflow(
                self.workflow_template,
                input_name,
                prompt,
                output_prefix,
                deterministic_seed(seed_key or frame_key),
                self.steps,
                self.lora_strength,
                # Distilled few-step models cannot add detail from a partly noised latent (the result is a blurred
                # copy of the source), so they always sample from full noise.
                1.0 if self.speed_lora_name else plan.denoise,
                plan.lora_name,
                self.upscale_model_name,
                style_input_name if style_reference is not None else None,
                latent_size_for(frame.size),
                plan.negative_prompt,
                self.speed_lora_name,
                SPEED_LORA_CFG if self.speed_lora_name else None,
            )
            submitted = self.request_json("/prompt", workflow)
            result = self.wait_for_history(submitted["prompt_id"])
            image_info = result["outputs"][UPSCALE_SAVE_NODE_ID]["images"][0]
            output_path = self.output_root / image_info.get("subfolder", "") / image_info["filename"]
            with Image.open(output_path) as opened:
                return opened.copy()
        finally:
            input_path.unlink(missing_ok=True)
            style_input_path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)
