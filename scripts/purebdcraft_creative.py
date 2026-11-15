"""PureBDCraft-oriented creative generation policy for Minecraft textures."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CREATIVE_TEXTURE_CATEGORIES = ("item", "block", "particle", "environment")
DEFAULT_CATEGORY_LORAS = {
    "block": "minecraft-block-qwen-edit-2511-r16.safetensors",
    "item": "minecraft-item-qwen-edit-2511-r16.safetensors",
}


@dataclass(frozen=True)
class TextureStyleProfile:
    denoise: float
    prompt_prefix: str
    prompt_constraints: str
    negative_prompt: str = ""
    use_style_reference: bool = True


@dataclass(frozen=True)
class CreativeGenerationPlan:
    category: str
    prompt: str
    denoise: float
    lora_name: str | None
    style_reference: Path | None
    negative_prompt: str = ""


STYLE_PROFILES: dict[str, TextureStyleProfile] = {
    "block": TextureStyleProfile(
        # Measured on raw model output: sampling from the source latent at 0.9 keeps the source layout,
        # silhouette and colours while still restyling; a second style image made the model copy its objects.
        denoise=0.9,
        use_style_reference=False,
        # Mirrors the caption the block LoRA was trained with (prepare_qwen_edit_training_dataset.py), plus an
        # explicit face-texture constraint: a block texture is a flat surface, never a depicted object.
        prompt_prefix=(
            "Creatively restyle this Minecraft block material named {name} as a seamless polished cartoon "
            "game texture."
        ),
        prompt_constraints=(
            "Preserve its material identity and make all edges tile continuously with matched colors and details. "
            "This is a flat block face texture: fill the entire frame edge to edge with the material surface, "
            "keeping the same occupancy map and major shape boundaries as the source. "
            "Do not add objects, scenery, text, pixel grids, borders, or directional shadows, and never draw "
            "an isolated object, container, centered emblem, or item."
        ),
        negative_prompt=(
            "isolated object, item icon, centered emblem, logo, container, bowl, scene, pixel grid, checkerboard, "
            "mosaic squares, blocky pixels, pixelated, jpeg artifacts, blur, noise, border, frame, text, watermark"
        ),
    ),
    "item": TextureStyleProfile(
        denoise=0.8,
        prompt_prefix=(
            "Redesign this Minecraft item texture named {name} in the PureBDCraft visual standard "
            "as a polished hand-drawn inventory icon."
        ),
        prompt_constraints=(
            "Preserve identity, silhouette, facing, proportions, and transparency. Use a bold dark contour, "
            "clean antialiased curves, broad color regions, controlled cel shading, and readable highlights. "
            "Keep the item centered, isolated, and readable at small size. Do not add scenery, hands, props, extra copies, text, stickers, "
            "blur, pixel noise, or a new background."
        ),
    ),
    "particle": TextureStyleProfile(
        denoise=0.5,
        prompt_prefix=(
            "Redesign this Minecraft particle texture named {name} in the PureBDCraft visual standard "
            "as a clean hand-drawn VFX sprite."
        ),
        prompt_constraints=(
            "Preserve effect identity, shape language, compact composition, and transparency. Use controlled "
            "soft edges, bold readable forms, and clean color separation. Keep the sprite simple, centered, readable, and compact. Do not add a scene, object sheet, "
            "text, border, or complex illustration."
        ),
    ),
    "environment": TextureStyleProfile(
        denoise=0.45,
        prompt_prefix=(
            "Redesign this Minecraft environment texture named {name} in the PureBDCraft visual standard "
            "as a seamless stylized game surface with hand-drawn detail."
        ),
        prompt_constraints=(
            "Preserve atmosphere, horizon flow, broad value structure, orientation, and tiling behavior. Use "
            "clean stylized gradients and controlled material breakup. Do not add text, isolated objects, "
            "posters, duplicate elements, or artificial borders."
        ),
    ),
    "texture": TextureStyleProfile(
        denoise=0.65,
        prompt_prefix=(
            "Redesign this Minecraft texture named {name} in the PureBDCraft visual standard "
            "as a crisp hand-drawn game texture."
        ),
        prompt_constraints=(
            "Preserve identity, layout, orientation, proportions, and transparency. Use bold clean outlines, "
            "broad color regions, controlled cel shading, and readable material highlights. Keep seamless assets "
            "tileable. Do not add scenery, text, borders, duplicate elements, blur, dithering, or pixel noise."
        ),
    ),
}


def texture_category(relative_path: Path) -> str | None:
    category = next((candidate for candidate in CREATIVE_TEXTURE_CATEGORIES if candidate in relative_path.parts), None)
    if category is not None:
        return category
    return None if {"entity", "gui"} & set(relative_path.parts) else "texture"


def prompt_for(relative_path: Path) -> str:
    category = texture_category(relative_path)
    if category is None:
        raise ValueError(f"Texture is not eligible for creative generation: {relative_path}")
    profile = STYLE_PROFILES[category]
    name = relative_path.stem.split("__", 1)[0].replace("_", " ").replace("-", " ")
    return f"{profile.prompt_prefix.format(name=name)} {profile.prompt_constraints}"


def denoise_for(relative_path: Path) -> float:
    category = texture_category(relative_path)
    return STYLE_PROFILES[category].denoise if category is not None else 0.0


def parse_category_loras(values: list[str]) -> dict[str, str]:
    category_loras = DEFAULT_CATEGORY_LORAS.copy()
    for value in values:
        category, separator, lora_name = value.partition("=")
        if separator != "=" or category not in CREATIVE_TEXTURE_CATEGORIES or not lora_name:
            allowed = ", ".join(CREATIVE_TEXTURE_CATEGORIES)
            raise ValueError(f"Invalid --category-lora {value!r}; expected one of {allowed}=FILENAME")
        category_loras[category] = lora_name
    return category_loras


def deterministic_seed(frame_key: str) -> int:
    return int.from_bytes(hashlib.sha256(frame_key.encode("utf-8")).digest()[:8], "big")


def style_reference_candidates(reference_root: Path, category: str) -> list[Path]:
    specific_dataset = {"item": "qwen_edit_pairs_item", "block": "qwen_edit_pairs_block"}.get(category)
    if specific_dataset is not None:
        target_root = reference_root / specific_dataset / "musubi_target"
        return sorted(path for path in target_root.glob("*.png") if path.is_file())

    metadata_path = reference_root / "qwen_edit_pairs" / "metadata.jsonl"
    if not metadata_path.is_file():
        return []
    marker = f" minecraft {category} named "
    candidates: list[Path] = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            record = json.loads(line)
            if marker not in f" {record.get('caption', '').casefold()} ":
                continue
            candidate = Path(record.get("image_path", ""))
            if candidate.is_file():
                candidates.append(candidate)
    return sorted(candidates)


def style_reference_pairs(reference_root: Path, category: str) -> list[tuple[Path, Path]]:
    specific_dataset = {"item": "qwen_edit_pairs_item", "block": "qwen_edit_pairs_block"}.get(category)
    if specific_dataset is not None:
        dataset_root = reference_root / specific_dataset
        control_root = dataset_root / "musubi_control"
        target_root = dataset_root / "musubi_target"
        return [
            (control_path, target_root / control_path.name)
            for control_path in sorted(control_root.glob("*.png"))
            if (target_root / control_path.name).is_file()
        ]

    metadata_path = reference_root / "qwen_edit_pairs" / "metadata.jsonl"
    if not metadata_path.is_file():
        return []
    marker = f" minecraft {category} named "
    pairs: list[tuple[Path, Path]] = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            record = json.loads(line)
            if marker not in f" {record.get('caption', '').casefold()} ":
                continue
            control_path = Path(record.get("control_path", ""))
            target_path = Path(record.get("image_path", ""))
            if control_path.is_file() and target_path.is_file():
                pairs.append((control_path, target_path))
    return pairs


def style_descriptor(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA").resize((16, 16), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3] * alpha[:, :, None]
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.clip(cv2.magnitude(edge_x, edge_y), 0.0, 1.0)
    return np.concatenate((rgb.reshape(-1), alpha.reshape(-1) * 2.0, edges.reshape(-1)))


def texture_name_tokens(value: str) -> set[str]:
    stem = Path(value).stem.split("__", 1)[-1]
    return {token for token in re.split(r"[^a-z0-9]+", stem.casefold()) if len(token) > 2}


class StyleReferenceIndex:
    def __init__(self, pairs: list[tuple[Path, Path]]) -> None:
        descriptors: list[np.ndarray] = []
        valid_pairs: list[tuple[Path, Path]] = []
        for control_path, target_path in pairs:
            try:
                with Image.open(control_path) as opened:
                    descriptors.append(style_descriptor(opened))
                valid_pairs.append((control_path, target_path))
            except (OSError, ValueError):
                continue
        self.pairs = valid_pairs
        self.target_tokens = [texture_name_tokens(target_path.name) for _, target_path in valid_pairs]
        self.descriptors = np.stack(descriptors) if descriptors else np.empty((0, 1280), dtype=np.float32)

    def nearest(self, source_image: Image.Image, frame_key: str) -> Path | None:
        if not self.pairs:
            return None
        query = style_descriptor(source_image)
        visual_distances = np.mean(np.abs(self.descriptors - query), axis=1)
        source_tokens = texture_name_tokens(frame_key)
        name_bonuses = np.array(
            [
                len(source_tokens & target_tokens) / max(1, len(source_tokens))
                for target_tokens in self.target_tokens
            ],
            dtype=np.float32,
        )
        return self.pairs[int(np.argmin(visual_distances - (name_bonuses * 0.2)))][1]


class PureBDCraftCreativeStrategy:
    def __init__(self, reference_root: Path | None, category_loras: dict[str, str]) -> None:
        self.reference_root = reference_root
        self.category_loras = category_loras
        self.reference_indexes: dict[str, StyleReferenceIndex] = {}
        self._index_lock = threading.Lock()
        self.reference_candidates: dict[str, list[Path]] = {}

    def plan_for(
        self,
        relative_path: Path,
        source_image: Image.Image,
        frame_key: str | None = None,
    ) -> CreativeGenerationPlan | None:
        category = texture_category(relative_path)
        if category is None:
            return None
        profile = STYLE_PROFILES[category]
        key = frame_key or relative_path.as_posix()
        return CreativeGenerationPlan(
            category=category,
            prompt=prompt_for(relative_path),
            denoise=profile.denoise,
            lora_name=self.category_loras.get(category),
            style_reference=(
                self._style_reference(category, key, source_image) if profile.use_style_reference else None
            ),
            negative_prompt=profile.negative_prompt,
        )

    def _style_reference(self, category: str, frame_key: str, source_image: Image.Image) -> Path | None:
        if self.reference_root is None:
            return None
        with self._index_lock:
            if category not in self.reference_indexes:
                self.reference_indexes[category] = StyleReferenceIndex(
                    style_reference_pairs(self.reference_root, category)
                )
        nearest = self.reference_indexes[category].nearest(source_image, frame_key)
        if nearest is not None:
            return nearest
        if category not in self.reference_candidates:
            self.reference_candidates[category] = style_reference_candidates(self.reference_root, category)
        candidates = self.reference_candidates[category]
        if not candidates:
            return None
        return candidates[deterministic_seed(frame_key) % len(candidates)]


def select_style_reference(
    reference_root: Path,
    category: str,
    frame_key: str,
    source_image: Image.Image | None = None,
) -> Path | None:
    strategy = PureBDCraftCreativeStrategy(reference_root, {})
    if source_image is not None:
        return strategy._style_reference(category, frame_key, source_image)
    candidates = style_reference_candidates(reference_root, category)
    if not candidates:
        return None
    return candidates[deterministic_seed(frame_key) % len(candidates)]
