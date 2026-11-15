"""Creative and structural frame assembly for Qwen-generated Minecraft textures."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from resourcepack_layout import copy_texture_metadata, target_size
from texture_frames import (
    animation_grid,
    assemble_animation_frames,
    load_texture_metadata,
    rgb_with_transparency_bleed,
    split_animation_frames,
)
from generate_pbr_maps import generate_for_texture, save_image_atomic
from purebdcraft_creative import prompt_for, texture_category
from qwen_cutout_keying import (
    background_clause,
    choose_key_colour,
    extract_cutout,
    flatten_on_key,
    is_binary_cutout,
)
from qwen_structural_processing import is_valid_image, load_source_image, upscale_structural_texture_if_needed
from remove_connected_background import connected_background_mask

RenderFrame = Callable[[Image.Image, str, str, str | None], Image.Image]
SOURCE_GUIDE_BLUR_PER_PIXEL = 0.6
SOURCE_GUIDE_MIX = 0.15
SOURCE_GUIDE_MIX_ANIMATED = 0.12
LOW_FREQUENCY_ANCHOR = 0.6
LOW_FREQUENCY_RADIUS_PER_PIXEL = 0.75
ANIMATION_MOTION_STRENGTH = 1.0
GRAYSCALE_CHANNEL_TOLERANCE = 2
CUTOUT_EDGE_BLUR_PER_PIXEL = 0.3
CUTOUT_EDGE_STEEPNESS = 12.0
MAX_SCROLL_DETECTION_SIZE = 64
SCROLL_MATCH_TOLERANCE = 1.0
TEXTURE_VARIANT_SUFFIXES = {
    "top", "bottom", "side", "side0", "side1", "side2", "side3", "front", "back", "left", "right",
    "north", "south", "east", "west", "up", "down", "inner", "outer", "end", "middle", "corner", "cap",
    "open", "closed", "on", "off", "lit", "unlit", "powered", "unpowered", "flow", "still", "upper",
    "lower", "triggered", "crafting", "glow", "overlay", "cracked",
}
TEXTURE_VARIANT_PATTERNS = (
    re.compile(r"^(?:stage|age|heat|growth|layer)\d+$"),
    re.compile(r"^\d+$"),
)


def texture_family_name(relative_path: Path) -> str:
    stem = relative_path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    while len(parts) > 1 and (
        parts[-1] in TEXTURE_VARIANT_SUFFIXES
        or any(pattern.match(parts[-1]) for pattern in TEXTURE_VARIANT_PATTERNS)
    ):
        parts.pop()
    return "_".join(parts) if parts else stem


def seed_key_for(relative_path: Path, frame_index: int, frame_count: int) -> str:
    family_name = texture_family_name(relative_path)
    if texture_category(relative_path) == "block" and family_name != relative_path.stem:
        return relative_path.with_name(family_name).as_posix()
    if frame_count > 1:
        # One seed per animation keeps independently rendered frames as close to each other as possible.
        return relative_path.as_posix()
    return f"{relative_path.as_posix()}__frame_{frame_index:04d}"


def has_internal_alpha_holes(frame: Image.Image) -> bool:
    if "A" not in frame.getbands():
        return False
    alpha = np.array(frame.convert("RGBA").getchannel("A"))
    transparent = alpha == 0
    if not transparent.any():
        return False
    labels_count, labels = cv2.connectedComponents(transparent.astype(np.uint8), connectivity=4)
    if labels_count <= 1:
        return False
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    return any(label not in border_labels for label in range(1, labels_count))


def internal_alpha_hole_mask(frame: Image.Image) -> Image.Image:
    alpha = np.array(frame.convert("RGBA").getchannel("A"))
    transparent = (alpha == 0).astype(np.uint8)
    labels_count, labels = cv2.connectedComponents(transparent, connectivity=4)
    if labels_count <= 1:
        return Image.new("L", frame.size, 0)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    hole_mask = np.zeros_like(alpha, dtype=np.uint8)
    for label in range(1, labels_count):
        if label not in border_labels:
            hole_mask[labels == label] = 255
    return Image.fromarray(hole_mask, mode="L")


def cleanup_transparent_rgb(image: Image.Image) -> Image.Image:
    rgba = np.array(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3:4]
    semi = (alpha > 0) & (alpha < 255)
    if np.any(semi):
        restored = (rgba[:, :, :3] - (255.0 - alpha)) * (255.0 / np.maximum(alpha, 1.0))
        rgba[:, :, :3] = np.where(semi, np.clip(restored, 0.0, 255.0), rgba[:, :, :3])
    rgba[:, :, :3] = np.where(alpha == 0, 0.0, rgba[:, :, :3])
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def finalize_frame(
    rendered: Image.Image,
    source: Image.Image,
    resolution: int,
    preserve_alpha_mask: bool = False,
) -> Image.Image:
    size = target_size(source.size, resolution)
    rendered_rgba = rendered.convert("RGBA")
    resized = rendered_rgba.resize(size, Image.Resampling.LANCZOS)
    source_alpha = source.convert("RGBA").getchannel("A") if "A" in source.getbands() else None
    if source_alpha is None:
        return resized.convert("RGB")
    hole_alpha_resized = None
    if preserve_alpha_mask:
        hole_alpha_resized = internal_alpha_hole_mask(source).resize(size, Image.Resampling.LANCZOS)
    if source_alpha.getextrema() == (255, 255):
        resized.putalpha(Image.new("L", size, 255))
        return resized
    rendered_array = np.asarray(rendered_rgba)
    background = connected_background_mask(rendered_array, minimum_lightness=145, maximum_chroma=60)
    generated_alpha = Image.fromarray(np.where(background, 0, 255).astype(np.uint8), mode="L")
    if generated_alpha.getextrema() == (255, 255):
        generated_alpha = source_alpha.resize(rendered_rgba.size, Image.Resampling.LANCZOS)
    alpha = generated_alpha.resize(size, Image.Resampling.LANCZOS)
    if preserve_alpha_mask and hole_alpha_resized is not None:
        alpha = ImageChops.subtract(alpha, hole_alpha_resized)
    resized.putalpha(alpha)
    return cleanup_transparent_rgb(resized)


def source_block_reference_frame(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Upscale the source as a smooth colour guide, so its pixel grid is never imprinted on the output."""
    upscaled = rgb_with_transparency_bleed(frame.convert("RGBA")).convert("RGB").resize(size, Image.Resampling.BICUBIC)
    pixel_scale = max(size[0] / max(1, frame.width), size[1] / max(1, frame.height))
    return upscaled.filter(ImageFilter.GaussianBlur(radius=max(1.0, pixel_scale * SOURCE_GUIDE_BLUR_PER_PIXEL)))


def animate_block_frame_from_anchor(
    anchor_frame: Image.Image,
    base_source: Image.Image,
    current_source: Image.Image,
) -> Image.Image:
    """Carry the source animation onto the anchor as a smooth colour delta, keeping the anchor's drawn detail."""
    anchor_array = np.asarray(anchor_frame.convert("RGB"), dtype=np.float32)
    base_array = np.asarray(base_source.convert("RGB"), dtype=np.float32)
    current_array = np.asarray(current_source.convert("RGB"), dtype=np.float32)
    animated = np.clip(anchor_array + (current_array - base_array) * ANIMATION_MOTION_STRENGTH, 0.0, 255.0)
    return Image.fromarray(animated.astype(np.uint8), mode="RGB")


def upscaled_source_alpha(frame: Image.Image, size: tuple[int, int]) -> Image.Image | None:
    """Upscale the source alpha; cutout (binary) masks get smooth antialiased edges instead of pixel stairs."""
    if "A" not in frame.getbands():
        return None
    alpha = frame.getchannel("A")
    upscaled = alpha.resize(size, Image.Resampling.BICUBIC)
    if sum(alpha.histogram()[1:255]) > 0:
        return upscaled
    pixel_scale = max(size[0] / max(1, frame.width), size[1] / max(1, frame.height))
    smoothed = upscaled.filter(ImageFilter.GaussianBlur(radius=max(0.5, pixel_scale * CUTOUT_EDGE_BLUR_PER_PIXEL)))
    return smoothed.point(lambda value: max(0, min(255, int((value - 128) * CUTOUT_EDGE_STEEPNESS + 128))))


def detect_cyclic_scroll(base_frame: Image.Image, current_frame: Image.Image) -> tuple[int, int] | None:
    """Return the (dx, dy) wrap-around shift that turns ``base_frame`` into ``current_frame``, if there is one."""
    if base_frame.size != current_frame.size or max(base_frame.size) > MAX_SCROLL_DETECTION_SIZE:
        return None
    base = np.asarray(base_frame.convert("RGBA"), dtype=np.int16)
    current = np.asarray(current_frame.convert("RGBA"), dtype=np.int16)
    if np.abs(current - base).mean() <= SCROLL_MATCH_TOLERANCE:
        return None
    height, width = base.shape[:2]
    for dy in range(height):
        for dx in range(width):
            if (dx or dy) and np.abs(np.roll(base, (dy, dx), axis=(0, 1)) - current).mean() <= SCROLL_MATCH_TOLERANCE:
                return dx, dy
    return None


def block_conditioning_frame(frame: Image.Image) -> Image.Image:
    """Flatten a block frame onto its mean material colour for the model.

    The model ignores alpha, so transparent pixels would otherwise reach it as arbitrary (usually black)
    colours and come back as dark fringes wherever its drawing and the source silhouette disagree.
    """
    rgba = frame.convert("RGBA")
    pixels = np.asarray(rgba, dtype=np.float32)
    weights = pixels[..., 3:4] / 255.0
    total = float(weights.sum())
    if total <= 0.0 or float(weights.min()) >= 1.0:
        return frame
    mean_colour = tuple(int(round(value)) for value in (pixels[..., :3] * weights).sum(axis=(0, 1)) / total)
    background = Image.new("RGBA", rgba.size, (*mean_colour, 255))
    return Image.alpha_composite(background, rgba)


def compose_block_frame(rgb: Image.Image, frame: Image.Image, resolution: int, animated: bool) -> Image.Image:
    """Refine a block frame and give it the transparency of its own source frame."""
    refined = refine_block_definition(rgb, frame, resolution, animated)
    alpha = upscaled_source_alpha(frame, refined.size)
    if alpha is not None:
        refined.putalpha(alpha)
    return refined


def anchor_low_frequency_to_source(generated: Image.Image, source_guide: Image.Image, pixel_scale: float) -> Image.Image:
    """Shift the local mean colour of the generated image towards the source, keeping its sharp detail.

    The model's drawing carries the detail; the source decides which colour and value belongs where. Only the
    difference between the low-pass versions is applied, so no detail is blurred and no source pixel grid
    is imprinted.
    """
    radius = max(2.0, pixel_scale * LOW_FREQUENCY_RADIUS_PER_PIXEL)
    low_generated = np.asarray(generated.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    low_source = np.asarray(source_guide.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    detailed = np.asarray(generated, dtype=np.float32) + (low_source - low_generated) * LOW_FREQUENCY_ANCHOR
    return Image.fromarray(np.clip(detailed, 0.0, 255.0).astype(np.uint8), mode="RGB")


def align_block_luminance_structure(
    generated_rgb: Image.Image,
    source_guide: Image.Image,
    animated: bool,
) -> Image.Image:
    source_luma = np.asarray(source_guide.convert("L"), dtype=np.float32) / 255.0
    generated = np.asarray(generated_rgb.convert("RGB"), dtype=np.float32)
    factor = 0.78 + source_luma[..., None] * (0.34 if animated else 0.42)
    return Image.fromarray(np.clip(generated * factor, 0.0, 255.0).astype(np.uint8), mode="RGB")


def finish_creative_block_quality(image: Image.Image, animated: bool) -> Image.Image:
    rgb = ImageEnhance.Contrast(image.convert("RGB")).enhance(1.06 if animated else 1.1)
    return rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110 if animated else 130, threshold=2))


def is_grayscale_source(frame: Image.Image) -> bool:
    """Grayscale sources are tinted by the game at runtime, so any hue added to them would corrupt the tint."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.int16)
    return int(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).max()) <= GRAYSCALE_CHANNEL_TOLERANCE


def refine_block_definition(
    image: Image.Image,
    source: Image.Image,
    resolution: int,
    animated: bool,
) -> Image.Image:
    """Pull the generated block back towards the source colour/value layout without hard source edges."""
    size = target_size(source.size, resolution)
    base_rgba = image.convert("RGBA")
    source_guide = source_block_reference_frame(source, size)
    pixel_scale = max(size[0] / max(1, source.width), size[1] / max(1, source.height))
    anchored_rgb = anchor_low_frequency_to_source(base_rgba.convert("RGB"), source_guide, pixel_scale)
    structured_rgb = align_block_luminance_structure(anchored_rgb, source_guide, animated)
    guided_rgb = Image.blend(structured_rgb, source_guide, SOURCE_GUIDE_MIX_ANIMATED if animated else SOURCE_GUIDE_MIX)
    refined_rgb = finish_creative_block_quality(guided_rgb, animated)
    if is_grayscale_source(source):
        refined_rgb = refined_rgb.convert("L").convert("RGB")
    output = refined_rgb.convert("RGBA")
    output.putalpha(base_rgba.getchannel("A"))
    return cleanup_transparent_rgb(output)


def render_consistent_animated_block_frames(
    frames: list[Image.Image],
    relative_path: Path,
    resolution: int,
    render_frame: RenderFrame,
) -> list[Image.Image]:
    frame_count = len(frames)
    anchor_source = frames[0]
    anchor_rendered = render_frame(
        block_conditioning_frame(anchor_source),
        prompt_for(relative_path),
        f"{relative_path.as_posix()}__frame_0000",
        seed_key_for(relative_path, 0, frame_count),
    )
    # One render defines the drawn detail; every frame reuses it, is refined exactly once and takes its
    # motion and transparency from its own source frame, so the animation cannot drift between frames.
    size = target_size(anchor_source.size, resolution)
    anchor_base = anchor_rendered.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    base_source = source_block_reference_frame(anchor_source, size)
    generated_frames: list[Image.Image] = []
    for frame in frames:
        scroll = detect_cyclic_scroll(anchor_source, frame)
        if scroll is not None:
            # A scrolling source (flowing liquids) moves the drawing itself, wrapped like the source.
            shift_x = round(scroll[0] * size[0] / anchor_source.width)
            shift_y = round(scroll[1] * size[1] / anchor_source.height)
            animated = ImageChops.offset(anchor_base, shift_x, shift_y)
        else:
            animated = animate_block_frame_from_anchor(
                anchor_base, base_source, source_block_reference_frame(frame, size)
            )
        generated_frames.append(compose_block_frame(animated, frame, resolution, animated=True))
    return generated_frames


def render_keyed_cutout_frame(
    frame: Image.Image,
    relative_path: Path,
    resolution: int,
    render_frame: RenderFrame,
    base_key: str,
    seed_key: str | None,
) -> Image.Image | None:
    """Redraw a binary cutout on a flat key colour and take its contour from the drawing, not the source.

    Returns None when keying did not produce a usable silhouette, so the caller can use the source-alpha path.
    """
    key = choose_key_colour(frame)
    prompt = f"{prompt_for(relative_path)} {background_clause(key)}"
    rendered = render_frame(flatten_on_key(frame, key), prompt, base_key, seed_key)
    size = target_size(frame.size, resolution)
    extracted = extract_cutout(rendered, frame, key.rgb, size)
    if extracted is None:
        return None
    rgb, alpha = extracted
    refined = refine_block_definition(rgb, frame, resolution, animated=False)
    refined.putalpha(alpha)
    return refined


def render_standard_texture_frames(
    frames: list[Image.Image],
    relative_path: Path,
    resolution: int,
    render_frame: RenderFrame,
    frame_count: int,
    is_block_texture: bool,
) -> list[Image.Image]:
    generated_frames: list[Image.Image] = []
    for frame_index, frame in enumerate(frames):
        base_key = f"{relative_path.as_posix()}__frame_{frame_index:04d}"
        if is_block_texture and is_binary_cutout(frame):
            keyed = render_keyed_cutout_frame(
                frame, relative_path, resolution, render_frame, base_key, seed_key_for(relative_path, frame_index, frame_count)
            )
            if keyed is not None:
                generated_frames.append(keyed)
                continue
        rendered = render_frame(
            block_conditioning_frame(frame) if is_block_texture else frame,
            prompt_for(relative_path),
            base_key,
            seed_key_for(relative_path, frame_index, frame_count),
        )
        if is_block_texture:
            # Blocks keep the full rendered surface and take their silhouette from the source alpha.
            size = target_size(frame.size, resolution)
            rendered_rgb = rendered.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            generated_frames.append(compose_block_frame(rendered_rgb, frame, resolution, animated=frame_count > 1))
            continue
        has_alpha = "A" in frame.getbands() and frame.convert("RGBA").getchannel("A").getextrema() != (255, 255)
        generated_frames.append(
            finalize_frame(
                rendered,
                frame,
                resolution,
                preserve_alpha_mask=has_alpha and has_internal_alpha_holes(frame),
            )
        )
    return generated_frames


def is_consistency_anchored_animated_block_texture(relative_path: Path, frames: list[Image.Image]) -> bool:
    return texture_category(relative_path) == "block" and len(frames) > 1


def resolve_items(
    discovered: dict[Path, Path],
    requested_paths: list[str],
    only_qwen: bool,
    only_category: str | None,
    reference_root: Path,
) -> list[tuple[Path, Path]]:
    if requested_paths:
        requested = [Path(relative_path) for relative_path in requested_paths]
        missing = [relative_path for relative_path in requested if relative_path not in discovered]
        if missing:
            raise FileNotFoundError(f"Textures not found: {', '.join(map(str, missing))}")
        items = [(relative_path, discovered[relative_path]) for relative_path in requested]
    else:
        items = sorted(discovered.items())
    if only_qwen:
        items = [(path, source) for path, source in items if texture_category(path) is not None]
    if only_category:
        items = [(path, source) for path, source in items if texture_category(path) == only_category]
    return [(path, source) for path, source in items if not (reference_root / path).is_file()]


def process_texture(
    source_path: Path,
    output_path: Path,
    output_root: Path,
    reference_root: Path,
    resolution: int,
    normal_strength: float,
    render_frame: RenderFrame,
    overwrite: bool,
) -> bool:
    relative_path = output_path.relative_to(output_root)
    structural_result = upscale_structural_texture_if_needed(
        relative_path,
        source_path,
        output_path,
        output_root,
        reference_root,
        resolution,
        normal_strength,
        overwrite,
    )
    if structural_result is not None:
        return structural_result
    if is_valid_image(output_path) and not overwrite:
        # Resume path: keep the albedo, but complete PBR maps lost to an interrupted run.
        generate_for_texture(output_path, output_root, reference_root, normal_strength, overwrite=False)
        return False
    source = load_source_image(source_path)
    metadata = load_texture_metadata(source_path)
    _, _, columns, rows = animation_grid(source, metadata)
    frames = split_animation_frames(source, metadata)
    frame_count = len(frames)
    if is_consistency_anchored_animated_block_texture(relative_path, frames):
        generated_frames = render_consistent_animated_block_frames(frames, relative_path, resolution, render_frame)
    else:
        generated_frames = render_standard_texture_frames(
            frames,
            relative_path,
            resolution,
            render_frame,
            frame_count,
            texture_category(relative_path) == "block",
        )
    output = assemble_animation_frames(generated_frames, columns, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_atomic(output, output_path, optimize=True)
    copy_texture_metadata(source_path, output_path)
    generate_for_texture(output_path, output_root, reference_root, normal_strength, overwrite=True)
    return True
