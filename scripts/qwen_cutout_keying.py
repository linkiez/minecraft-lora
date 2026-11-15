"""Chroma-key background handling for cutout (binary alpha) block textures.

The model ignores alpha, so a cutout is flattened on a single key colour, redrawn by the model on that flat
background, and the background is then removed by script. The drawn contour therefore follows the drawing
instead of the low-resolution source silhouette, which only limits where content may appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from texture_frames import rgb_with_transparency_bleed

KEY_COLOURS: dict[str, tuple[int, int, int]] = {
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "green": (0, 255, 0),
}
OPAQUE_ALPHA_THRESHOLD = 127
KEY_DISTANCE_PERCENTILE = 5
# Colour distance (0-441) below which a pixel is pure background and above which it is fully object.
KEY_DISTANCE_LOW = 70.0
KEY_DISTANCE_HIGH = 140.0
# How far, in source pixels, the drawing may extend beyond / must stay inside the source silhouette.
SILHOUETTE_GROW_PER_SOURCE_PIXEL = 1.0
# The hard source mask is a staircase at output size; blurring then re-steepening rounds it into a smooth contour.
SILHOUETTE_SMOOTHING_PER_SOURCE_PIXEL = 1.0
SILHOUETTE_SMOOTHING_STEEPNESS = 4.0
# An enclosed background-coloured region is a model blemish, not a window, when this much of it is source-opaque.
ENCLOSED_SPECK_SOURCE_FRACTION = 0.5
MIN_KEYED_COVERAGE = 0.35
# Only pixels this solid keep their own colour; edge pixels borrow it from the nearest solid neighbour.
SOLID_ALPHA_THRESHOLD = 0.98


@dataclass(frozen=True)
class KeyColour:
    """A fully saturated flat colour used as the temporary background of a cutout."""

    name: str
    rgb: tuple[int, int, int]

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


def is_binary_cutout(frame: Image.Image) -> bool:
    """True for sources whose alpha is only 0 or 255 with both present; partial alpha must keep its own path."""
    if "A" not in frame.getbands():
        return False
    alpha = frame.convert("RGBA").getchannel("A")
    lowest, highest = alpha.getextrema()
    return lowest == 0 and highest == 255 and sum(alpha.histogram()[1:255]) == 0


def choose_key_colour(frame: Image.Image) -> KeyColour:
    """Pick the key colour that sits furthest from the opaque colours of the object."""
    pixels = np.asarray(frame.convert("RGBA"), dtype=np.float32)
    object_colours = pixels[pixels[..., 3] > OPAQUE_ALPHA_THRESHOLD][:, :3]
    names = list(KEY_COLOURS)
    if len(object_colours) == 0:
        return KeyColour(names[0], KEY_COLOURS[names[0]])

    def clearance(name: str) -> float:
        distances = np.linalg.norm(object_colours - np.array(KEY_COLOURS[name], dtype=np.float32), axis=1)
        return float(np.percentile(distances, KEY_DISTANCE_PERCENTILE))

    best = max(names, key=clearance)
    return KeyColour(best, KEY_COLOURS[best])


def flatten_on_key(frame: Image.Image, key: KeyColour) -> Image.Image:
    """Composite a cutout on the key colour, which is what the model receives instead of transparency."""
    background = Image.new("RGBA", frame.size, (*key.rgb, 255))
    return Image.alpha_composite(background, frame.convert("RGBA")).convert("RGB")


def background_clause(key: KeyColour) -> str:
    """Prompt addition asking the model to keep the background one flat key colour."""
    return (
        f"The background is one flat solid {key.name} colour {key.hex}, with no shading, no gradient, no texture "
        "and no shadow; the object itself never uses that colour."
    )


def key_alpha(rendered: Image.Image, key: tuple[int, int, int]) -> np.ndarray:
    """Alpha in 0..1 from the colour distance to the key: 0 on the flat background, 1 on the drawn object."""
    pixels = np.asarray(rendered.convert("RGB"), dtype=np.float32)
    distance = np.linalg.norm(pixels - np.array(key, dtype=np.float32), axis=2)
    return np.clip((distance - KEY_DISTANCE_LOW) / (KEY_DISTANCE_HIGH - KEY_DISTANCE_LOW), 0.0, 1.0)


def _silhouette_masks(frame: Image.Image, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return the smooth grown silhouette (float) and the plain source mask (bool) at the output size."""
    source_alpha = frame.convert("RGBA").getchannel("A")
    mask = (np.asarray(source_alpha.resize(size, Image.Resampling.NEAREST)) > OPAQUE_ALPHA_THRESHOLD).astype(np.uint8)
    pixel_scale = max(size[0] / max(1, frame.width), size[1] / max(1, frame.height))
    width = max(1, int(round(pixel_scale * SILHOUETTE_GROW_PER_SOURCE_PIXEL))) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    grown = cv2.dilate(mask, kernel).astype(np.float32)
    blurred = cv2.GaussianBlur(grown, (0, 0), pixel_scale * SILHOUETTE_SMOOTHING_PER_SOURCE_PIXEL)
    smooth = np.clip((blurred - 0.5) * SILHOUETTE_SMOOTHING_STEEPNESS + 0.5, 0.0, 1.0)
    return smooth, mask.astype(bool)


def _fill_enclosed_specks(alpha: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    """Make enclosed transparent regions opaque when the source is solid there (model painted background inside)."""
    labels_count, labels = cv2.connectedComponents((alpha < 0.5).astype(np.uint8), connectivity=4)
    if labels_count <= 1:
        return alpha
    flat_labels = labels.ravel()
    totals = np.bincount(flat_labels, minlength=labels_count).astype(np.float64)
    inside = np.bincount(flat_labels, weights=source_mask.ravel().astype(np.float64), minlength=labels_count)
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    fill = inside / np.maximum(totals, 1.0) > ENCLOSED_SPECK_SOURCE_FRACTION
    fill[0] = False
    fill[border] = False
    return np.where(fill[labels], 1.0, alpha)


def extract_cutout(
    rendered: Image.Image,
    frame: Image.Image,
    key: tuple[int, int, int],
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image] | None:
    """Separate a key-coloured render into key-free RGB and an alpha mask limited to the source silhouette.

    The drawn contour decides the edges, the grown source silhouette discards anything invented outside it,
    and enclosed background-coloured specks inside the source silhouette are filled. Returns
    None when too little of the silhouette survived keying, so the caller can fall back to the source alpha.
    """
    resized = rendered.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    keyed_alpha = key_alpha(resized, key)
    grown, source_mask = _silhouette_masks(frame, size)
    alpha = np.minimum(_fill_enclosed_specks(keyed_alpha, source_mask), grown)

    silhouette_area = float(grown.sum())
    if silhouette_area <= 0.0 or float((keyed_alpha * grown).sum()) / silhouette_area < MIN_KEYED_COVERAGE:
        return None

    # Edge colours come from solid drawn pixels, so no key colour can reach a pixel that stays visible.
    solid = np.where(keyed_alpha >= SOLID_ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    bleeding = np.dstack([np.asarray(resized, dtype=np.uint8), solid])
    rgb = rgb_with_transparency_bleed(Image.fromarray(bleeding, mode="RGBA"))
    return rgb, Image.fromarray(np.rint(alpha * 255.0).astype(np.uint8), mode="L")
