import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_pbr_maps import generate_normal_map, generate_specular_map, is_albedo_path, save_image_atomic


def test_flat_texture_generates_neutral_normal() -> None:
    albedo = Image.new("RGB", (32, 16), (128, 128, 128))

    normal = generate_normal_map(albedo)

    pixel = np.asarray(normal)[8, 16]
    assert normal.size == albedo.size
    assert tuple(pixel) == (128, 128, 255)


def test_specular_map_preserves_source_alpha() -> None:
    albedo = Image.new("RGBA", (8, 8), (220, 180, 80, 37))

    specular = generate_specular_map(albedo, Path("textures/block/gold_block.png"))

    assert specular.mode == "RGBA"
    assert specular.getchannel("A").getextrema() == (37, 37)


def test_existing_pbr_maps_are_not_treated_as_albedo() -> None:
    assert is_albedo_path(Path("stone.png"))
    assert not is_albedo_path(Path("stone_n.png"))
    assert not is_albedo_path(Path("stone_s.png"))

def test_save_image_atomic_uses_umask_based_permissions(tmp_path: Path) -> None:
    target = tmp_path / "out" / "texture.png"

    save_image_atomic(Image.new("RGB", (4, 4), "red"), target)

    current_umask = os.umask(0)
    os.umask(current_umask)
    assert (target.stat().st_mode & 0o777) == (0o666 & ~current_umask)
