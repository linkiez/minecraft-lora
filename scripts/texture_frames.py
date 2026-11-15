"""Helpers for animated texture sheets and transparency-safe RGB handling."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_texture_metadata(texture_path: Path) -> dict:
    """Return the parsed ``.mcmeta`` file next to a texture, or an empty dict when it is missing or invalid."""
    metadata_path = Path(f"{texture_path}.mcmeta")
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def animation_grid(image: Image.Image, metadata: dict) -> tuple[int, int, int, int]:
    """Return ``(frame_width, frame_height, columns, rows)`` for a texture, treating non-animations as one frame."""
    animation = metadata.get("animation")
    if not isinstance(animation, dict):
        return image.width, image.height, 1, 1
    frame_width = int(animation.get("width", image.width))
    frame_height = int(animation.get("height", frame_width))
    if frame_width <= 0 or frame_height <= 0:
        return image.width, image.height, 1, 1
    columns = image.width // frame_width
    rows = image.height // frame_height
    if columns < 1 or rows < 1:
        return image.width, image.height, 1, 1
    return frame_width, frame_height, columns, rows


def split_animation_frames(image: Image.Image, metadata: dict) -> list[Image.Image]:
    """Cut an animation sheet into its frames in reading order."""
    frame_width, frame_height, columns, rows = animation_grid(image, metadata)
    return [
        image.crop((column * frame_width, row * frame_height, (column + 1) * frame_width, (row + 1) * frame_height))
        for row in range(rows)
        for column in range(columns)
    ]


def assemble_animation_frames(frames: list[Image.Image], columns: int, rows: int) -> Image.Image:
    """Paste frames back into a ``columns`` x ``rows`` sheet; the frame count must match the grid."""
    if not frames or len(frames) != columns * rows:
        raise ValueError("Animation frame count does not match its grid.")
    frame_width, frame_height = frames[0].size
    output = Image.new(frames[0].mode, (frame_width * columns, frame_height * rows))
    for index, frame in enumerate(frames):
        output.paste(frame, ((index % columns) * frame_width, (index // columns) * frame_height))
    return output


def rgb_with_transparency_bleed(image: Image.Image) -> Image.Image:
    """Return RGB where fully transparent pixels take the colour of the nearest opaque pixel.

    Resizing or filtering such an image never mixes in the arbitrary colour stored under transparent pixels.
    """
    if "A" not in image.getbands():
        return image.convert("RGB")

    rgba = np.asarray(image.convert("RGBA"))
    rgb = rgba[:, :, :3].copy()
    transparent = rgba[:, :, 3] == 0
    opaque = ~transparent
    if transparent.any() and opaque.any():
        _, labels = cv2.distanceTransformWithLabels(
            transparent.astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        palette = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        palette[labels[opaque]] = rgb[opaque]
        rgb[transparent] = palette[labels[transparent]]
    return Image.fromarray(rgb)
