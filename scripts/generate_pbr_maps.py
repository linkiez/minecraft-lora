#!/usr/bin/env python3
"""Generate LabPBR 1.3 normal and specular maps, estimated from the albedo, for a resource pack."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from texture_frames import (
    animation_grid,
    assemble_animation_frames,
    load_texture_metadata,
    split_animation_frames,
)

# This pipeline re-opens locally generated 4k animation sheets that can exceed
# Pillow's conservative decompression-bomb threshold while still being expected.
Image.MAX_IMAGE_PIXELS = None

# Constants follow the LabPBR 1.3 material standard (https://shaderlabs.org/wiki/LabPBR_Material_Standard).
# Specular green channel: 0-229 stores linear F0 (reflectance), 230-254 select a predefined metal, 255 is a generic
# metal that uses the albedo as F0.
DIELECTRIC_F0 = 10  # about 4 %, typical of non-metals
GENERIC_METAL = 255
METAL_IDS = {
    "iron": 230, "steel": 230, "anvil": 230, "chain": 230,
    "gold": 231, "aluminum": 232, "aluminium": 232, "chrome": 233,
    "copper": 234, "lead": 235, "platinum": 236, "silver": 237,
}
GENERIC_METAL_TERMS = {"metal", "bronze", "brass"}
# A texture of stone with metal specks, or of a raw lump, must not be shaded as solid metal.
NOT_SOLID_METAL_TERMS = {"ore", "raw"}
# Specular blue channel: 0-64 is porosity (the LabPBR examples: sand 64, wool 38, wood 12).
POROSITY_BY_TERM = {"sand": 64, "wool": 38, "carpet": 38, "planks": 12, "log": 12, "wood": 12, "stem": 12}
# Specular alpha: 0-254 is emission strength, 255 means "does not emit".
NO_EMISSION = 255
MAX_EMISSION = 254
EMISSIVE_TERMS = {
    "emissive", "glow", "glowing", "lava", "fire", "flame", "lantern", "torch", "light", "magma", "redstone"
}
# Normal alpha is height: 255 is the surface, lower values are deeper. The depth range actually used is limited so
# the parallax effect stays moderate, and 0 is avoided because some shader packs mishandle it.
HEIGHT_DEPTH_RANGE = 0.6
MIN_HEIGHT = 1
# Normal blue is ambient occlusion: 255 is none. Pixels darker than their surroundings count as cavities.
AO_STRENGTH = 0.5
AO_CAVITY_SCALE = 0.25
AO_BLUR_PER_SIDE = 0.06


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("texture"))
    parser.add_argument("--normal-strength", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_albedo_path(path: Path) -> bool:
    return path.suffix.lower() == ".png" and not path.stem.endswith(("_n", "_s"))


def height_field(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=1.1, sigmaY=1.1)


def height_channel(height: np.ndarray) -> np.ndarray:
    """LabPBR height (normal alpha): 255 at the highest point of the texture, lower where it is deeper."""
    depth = (float(height.max()) - height) * HEIGHT_DEPTH_RANGE
    return np.clip(np.rint(255.0 - depth * 255.0), MIN_HEIGHT, 255)


def ambient_occlusion_channel(height: np.ndarray) -> np.ndarray:
    """LabPBR material occlusion (normal blue): 255 on flat or raised areas, darker inside cavities."""
    sigma = max(3.0, AO_BLUR_PER_SIDE * max(height.shape))
    surroundings = cv2.GaussianBlur(height, (0, 0), sigmaX=sigma, sigmaY=sigma)
    cavity = np.clip((surroundings - height) / AO_CAVITY_SCALE, 0.0, 1.0)
    return np.clip(np.rint(255.0 * (1.0 - AO_STRENGTH * cavity)), 0, 255)


def generate_normal_map(image: Image.Image, strength: float = 2.0) -> Image.Image:
    """Build a LabPBR normal texture: X/Y in red/green (DirectX, Y down), occlusion in blue, height in alpha."""
    height = height_field(image)
    gradient_x = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.dstack((-gradient_x * strength, -gradient_y * strength, np.ones_like(height)))
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
    xy = np.clip(np.rint((normal[..., :2] * 0.5 + 0.5) * 255.0), 0, 255)
    encoded = np.dstack((xy, ambient_occlusion_channel(height), height_channel(height))).astype(np.uint8)
    return Image.fromarray(encoded, mode="RGBA")


def metal_value(words: set[str]) -> int | None:
    """The LabPBR green-channel metal id for a texture name, or None when it is not a solid metal."""
    if words & NOT_SOLID_METAL_TERMS:
        return None
    known = next((METAL_IDS[word] for word in sorted(words) if word in METAL_IDS), None)
    if known is not None:
        return known
    return GENERIC_METAL if words & GENERIC_METAL_TERMS else None


def material_channels(image: Image.Image, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the four LabPBR specular channels (smoothness, F0/metal, porosity, emission) from albedo and name."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)[:, :, 2] / 255.0
    detail = np.abs(cv2.Laplacian(value, cv2.CV_32F))
    smoothness = np.clip(72.0 + value * 72.0 - detail * 120.0, 0.0, 255.0)
    words = set(path.stem.lower().replace("-", "_").split("_"))
    metal = metal_value(words)
    f0 = np.full_like(value, DIELECTRIC_F0 if metal is None else metal)
    porosity_value = 0 if metal is not None else max((POROSITY_BY_TERM.get(word, 0) for word in words), default=0)
    porosity = np.full_like(value, porosity_value)
    emission_strength = np.clip(value * 220.0, 0.0, MAX_EMISSION)
    emissive = emission_strength if words & EMISSIVE_TERMS else np.full_like(value, NO_EMISSION)
    return smoothness, f0, porosity, emissive


def generate_specular_map(image: Image.Image, path: Path) -> Image.Image:
    """Build a LabPBR specular texture: smoothness, F0 or metal id, porosity, and emission in alpha."""
    channels = np.dstack(material_channels(image, path))
    return Image.fromarray(np.clip(np.rint(channels), 0, 255).astype(np.uint8), mode="RGBA")


def resize_reference(reference: Path, size: tuple[int, int], mode: str) -> Image.Image:
    with Image.open(reference) as image:
        return image.convert(mode).resize(size, Image.Resampling.LANCZOS)


def map_path(albedo_path: Path, suffix: str) -> Path:
    return albedo_path.with_name(f"{albedo_path.stem}{suffix}.png")


def reference_path(reference_root: Path, pack_root: Path, target: Path) -> Path:
    relative = target.relative_to(pack_root)
    return reference_root / relative


def _read_process_umask() -> int:
    """Read the umask once at import; os.umask can only be read by setting it, which is not thread-safe."""
    current = os.umask(0)
    os.umask(current)
    return current


PROCESS_UMASK = _read_process_umask()
MAX_PBR_SHEET_PIXELS = 4096 * 4096 * 8  # skip very large animated sheets to avoid hangs


def save_image_atomic(image: Image.Image, path: Path, **save_kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        image.save(temp_path, **save_kwargs)
        with Image.open(temp_path) as written:
            written.verify()
        # mkstemp creates 0600 files; restore the umask-based mode a normal write would have.
        os.chmod(temp_path, 0o666 & ~PROCESS_UMASK)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def generate_for_texture(
    albedo_path: Path,
    pack_root: Path,
    reference_root: Path,
    normal_strength: float,
    overwrite: bool,
) -> int:
    normal_path = map_path(albedo_path, "_n")
    specular_path = map_path(albedo_path, "_s")
    if not overwrite and normal_path.exists() and specular_path.exists():
        return 0
    with Image.open(albedo_path) as image:
        albedo = image.copy()
    if albedo.width * albedo.height > MAX_PBR_SHEET_PIXELS:
        return 0
    metadata = load_texture_metadata(albedo_path)
    _, _, columns, rows = animation_grid(albedo, metadata)
    frames = split_animation_frames(albedo, metadata)

    normal_reference = reference_path(reference_root, pack_root, normal_path)
    if overwrite or not normal_path.exists():
        if normal_reference.is_file():
            normal = resize_reference(normal_reference, albedo.size, "RGB")
        else:
            normal = assemble_animation_frames(
                [generate_normal_map(frame, normal_strength) for frame in frames], columns, rows
            )
        save_image_atomic(normal, normal_path)

    specular_reference = reference_path(reference_root, pack_root, specular_path)
    if overwrite or not specular_path.exists():
        if specular_reference.is_file():
            specular = resize_reference(specular_reference, albedo.size, "RGBA")
        else:
            specular = assemble_animation_frames(
                [generate_specular_map(frame, albedo_path) for frame in frames], columns, rows
            )
        save_image_atomic(specular, specular_path)

    metadata_path = Path(f"{albedo_path}.mcmeta")
    if metadata_path.is_file():
        shutil.copy2(metadata_path, Path(f"{normal_path}.mcmeta"))
        shutil.copy2(metadata_path, Path(f"{specular_path}.mcmeta"))
    return 1


def main() -> None:
    args = parse_args()
    assets_root = args.pack_root / "assets"
    generated = 0
    failed = 0
    for albedo_path in sorted(assets_root.glob("**/textures/**/*.png")):
        if not is_albedo_path(albedo_path):
            continue
        try:
            generated += generate_for_texture(
                albedo_path,
                args.pack_root,
                args.reference_root,
                args.normal_strength,
                args.overwrite,
            )
        except OSError as error:
            failed += 1
            print(f"PBR failed: {albedo_path}: {error}", flush=True)
    print(f"PBR complete: generated={generated} failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
