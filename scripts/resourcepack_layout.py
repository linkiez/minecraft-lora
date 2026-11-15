"""Resource-pack layout helpers: texture discovery, sizing and pack metadata."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from generate_pbr_maps import is_albedo_path

PACK_ICON_NAME = "pack.png"
PACK_FORMAT = 34


def discover_textures(source_root: Path) -> dict[Path, Path]:
    """Return pack-relative texture paths, with mods overriding vanilla assets."""
    textures: dict[Path, Path] = {}
    asset_roots = (source_root / "original" / "assets", source_root / "mods" / "assets")
    for assets_root in asset_roots:
        if not assets_root.is_dir():
            continue
        for source_path in sorted(assets_root.glob("**/textures/**/*.png")):
            if is_albedo_path(source_path):
                relative_path = Path("assets") / source_path.relative_to(assets_root)
                textures[relative_path] = source_path
    return textures


def target_size(source_size: tuple[int, int], resolution: int) -> tuple[int, int]:
    """Fit an image inside a square bound while preserving its aspect ratio."""
    width, height = source_size
    scale = resolution / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def copy_texture_metadata(source_path: Path, output_path: Path) -> None:
    """Copy the ``.mcmeta`` file of a texture next to its output, when the source has one."""
    metadata_path = Path(f"{source_path}.mcmeta")
    if metadata_path.is_file():
        shutil.copy2(metadata_path, Path(f"{output_path}.mcmeta"))


def copy_pack_icon(source_root: Path, output_root: Path) -> None:
    """Copy ``pack.png`` from ``texture/`` or the project root into the generated pack, if one exists."""
    for candidate in (source_root / "texture" / PACK_ICON_NAME, source_root / PACK_ICON_NAME):
        if candidate.is_file():
            shutil.copy2(candidate, output_root / PACK_ICON_NAME)
            return


def write_pack_metadata(output_root: Path, resolution: int) -> None:
    """Write ``pack.mcmeta`` for the generated pack."""
    metadata = {
        "pack": {
            "pack_format": PACK_FORMAT,
            "description": f"Artistic Smooth {resolution}px LabPBR - Mods + Original (1.21.1)",
        }
    }
    (output_root / "pack.mcmeta").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
