"""Structural processing for Minecraft GUI and entity textures."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from resourcepack_layout import copy_texture_metadata, target_size
from texture_frames import (
    animation_grid,
    assemble_animation_frames,
    load_texture_metadata,
    rgb_with_transparency_bleed,
    split_animation_frames,
)
from generate_pbr_maps import generate_for_texture, save_image_atomic


def load_source_image(path: Path) -> Image.Image:
    """Load a texture, expanding palette/colour-key transparency into a real alpha channel."""
    with Image.open(path) as opened:
        image = opened.copy()
    if "transparency" in image.info:
        image = image.convert("RGBA")
    return image


def is_valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def is_gui_texture(relative_path: Path) -> bool:
    parts = relative_path.parts
    return "textures" in parts and "gui" in parts[parts.index("textures") + 1 :]


def is_entity_texture(relative_path: Path) -> bool:
    parts = relative_path.parts
    return "textures" in parts and "entity" in parts[parts.index("textures") + 1 :]


def remove_gui_pbr(output_path: Path) -> None:
    for suffix in ("_n", "_s"):
        pbr_path = output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")
        pbr_path.unlink(missing_ok=True)
        Path(f"{pbr_path}.mcmeta").unlink(missing_ok=True)


def vectorize_structural_frame(frame: Image.Image, resolution: int, palette_colors: int = 16) -> Image.Image:
    size = target_size(frame.size, resolution)
    source_rgba = frame.convert("RGBA")
    quantized = rgb_with_transparency_bleed(source_rgba).quantize(
        colors=palette_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    ).convert("RGB")
    pixels = np.asarray(quantized)
    colors, labels = np.unique(pixels.reshape(-1, 3), axis=0, return_inverse=True)
    labels = labels.reshape(pixels.shape[:2])
    sigma_x = max(1.0, size[0] / frame.width * 0.35)
    sigma_y = max(1.0, size[1] / frame.height * 0.35)
    # Running argmax keeps a single full-resolution score plane alive instead of one per palette colour.
    best_score: np.ndarray | None = None
    output_labels = np.zeros((size[1], size[0]), dtype=np.intp)
    for color_index in range(len(colors)):
        mask = (labels == color_index).astype(np.float32)
        resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        score = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        if best_score is None:
            best_score = score
            continue
        better = score > best_score
        output_labels[better] = color_index
        best_score[better] = score[better]
    output = Image.fromarray(colors[output_labels].astype(np.uint8), mode="RGB")
    if "A" in frame.getbands():
        output = output.convert("RGBA")
        output.putalpha(source_rgba.getchannel("A").resize(size, Image.Resampling.LANCZOS))
    return output


def upscale_structural_texture(
    source_path: Path,
    output_path: Path,
    output_root: Path,
    reference_root: Path,
    resolution: int,
    normal_strength: float,
    overwrite: bool,
    generate_pbr: bool,
) -> bool:
    if is_valid_image(output_path) and not overwrite:
        if generate_pbr:
            generate_for_texture(output_path, output_root, reference_root, normal_strength, overwrite=False)
        else:
            remove_gui_pbr(output_path)
        return False
    source = load_source_image(source_path)
    metadata = load_texture_metadata(source_path)
    _, _, columns, rows = animation_grid(source, metadata)
    frames = split_animation_frames(source, metadata)
    output_frames = [vectorize_structural_frame(frame, resolution) for frame in frames]
    styled = assemble_animation_frames(output_frames, columns, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_atomic(styled, output_path, optimize=True)
    copy_texture_metadata(source_path, output_path)
    if generate_pbr:
        generate_for_texture(output_path, output_root, reference_root, normal_strength, overwrite=True)
    else:
        remove_gui_pbr(output_path)
    return True


def upscale_entity_texture(
    source_path: Path,
    output_path: Path,
    output_root: Path,
    reference_root: Path,
    resolution: int,
    normal_strength: float,
    overwrite: bool,
) -> bool:
    return upscale_structural_texture(
        source_path,
        output_path,
        output_root,
        reference_root,
        resolution,
        normal_strength,
        overwrite,
        generate_pbr=True,
    )


def upscale_structural_texture_if_needed(
    relative_path: Path,
    source_path: Path,
    output_path: Path,
    output_root: Path,
    reference_root: Path,
    resolution: int,
    normal_strength: float,
    overwrite: bool,
) -> bool | None:
    if is_gui_texture(relative_path):
        if not is_valid_image(source_path):
            unchanged = output_path.is_file() and output_path.read_bytes() == source_path.read_bytes()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not unchanged:
                shutil.copy2(source_path, output_path)
            copy_texture_metadata(source_path, output_path)
            remove_gui_pbr(output_path)
            return not unchanged
        return upscale_structural_texture(
            source_path,
            output_path,
            output_root,
            reference_root,
            resolution,
            normal_strength,
            overwrite,
            generate_pbr=False,
        )
    if is_entity_texture(relative_path):
        return upscale_entity_texture(
            source_path,
            output_path,
            output_root,
            reference_root,
            resolution,
            normal_strength,
            overwrite,
        )
    return None
