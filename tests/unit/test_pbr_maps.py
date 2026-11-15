import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_pbr_maps import (
    DIELECTRIC_F0,
    generate_normal_map,
    generate_specular_map,
    is_albedo_path,
    save_image_atomic,
)


def gradient_albedo(horizontal: bool) -> Image.Image:
    ramp = np.tile(np.linspace(40, 220, 32, dtype=np.uint8), (32, 1))
    return Image.fromarray(ramp if horizontal else ramp.T).convert("RGB")


def pitted_albedo() -> Image.Image:
    albedo = Image.new("RGB", (32, 32), (200, 200, 200))
    albedo.paste(Image.new("RGB", (8, 8), (20, 20, 20)), (12, 12))
    return albedo


def test_flat_texture_generates_a_neutral_labpbr_normal() -> None:
    albedo = Image.new("RGB", (32, 16), (128, 128, 128))

    normal = generate_normal_map(albedo)

    assert normal.mode == "RGBA"
    assert normal.size == albedo.size
    # X/Y are neutral, the blue channel is ambient occlusion (255 = none), alpha is height (255 = no depth).
    assert tuple(np.asarray(normal)[8, 16]) == (128, 128, 255, 255)


def test_normal_map_uses_the_directx_orientation_of_labpbr() -> None:
    rising_right = np.asarray(generate_normal_map(gradient_albedo(horizontal=True)))[16, 16]
    rising_down = np.asarray(generate_normal_map(gradient_albedo(horizontal=False)))[16, 16]

    # Red: 0 = normal points left; a surface that rises to the right leans left.
    assert rising_right[0] < 128
    # Green: 0 = normal points up (Y-); a surface that rises towards the bottom leans up.
    assert rising_down[1] < 128


def test_normal_map_stores_height_in_alpha_and_cavity_occlusion_in_blue() -> None:
    normal = np.asarray(generate_normal_map(pitted_albedo()))

    top, pit = normal[3, 3], normal[16, 16]
    assert pit[3] < top[3] == 255          # the dark pit is deeper than the surface
    assert pit[2] < 255 and top[2] == 255  # and occluded, while the flat surface is not
    assert normal[..., 3].min() >= 1       # LabPBR advises against a height of 0


def specular(name: str, colour: tuple[int, int, int] = (150, 150, 150)) -> np.ndarray:
    albedo = Image.new("RGB", (8, 8), colour)
    return np.asarray(generate_specular_map(albedo, Path(f"textures/block/{name}.png")))[4, 4]


def test_specular_map_is_rgba_with_labpbr_channels() -> None:
    image = generate_specular_map(Image.new("RGBA", (8, 8), (220, 180, 80, 37)), Path("textures/block/stone.png"))

    assert image.mode == "RGBA"
    smoothness, f0, porosity, emissive = np.asarray(image)[4, 4]
    assert 0 <= smoothness <= 255
    assert f0 == DIELECTRIC_F0 <= 229   # a dielectric stores its reflectance, not a metal id
    assert porosity == 0
    assert emissive == 255              # alpha is emissive, where 255 means "no emission"; source alpha is not copied


def test_predefined_metals_use_the_labpbr_metal_ids() -> None:
    assert specular("iron_block")[1] == 230
    assert specular("gold_block")[1] == 231
    assert specular("copper_block")[1] == 234
    assert specular("silver_ingot_block")[1] == 237
    assert specular("brass_plate_metal")[1] == 255  # a generic metal uses the albedo as F0
    assert specular("iron_block")[2] == 0           # metals are not porous


def test_ores_and_raw_blocks_are_not_treated_as_solid_metal() -> None:
    assert specular("iron_ore")[1] <= 229
    assert specular("raw_copper_block")[1] <= 229


def test_porosity_follows_the_labpbr_examples() -> None:
    assert specular("sand")[2] == 64
    assert specular("white_wool")[2] == 38
    assert specular("oak_planks")[2] == 12
    assert specular("stone")[2] == 0


def test_emissive_textures_set_alpha_below_255_and_others_leave_it_at_255() -> None:
    lava = specular("lava_still", (240, 120, 30))
    stone = specular("stone")

    assert 0 < lava[3] <= 254
    assert stone[3] == 255


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
