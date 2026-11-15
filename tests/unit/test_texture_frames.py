import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from texture_frames import (
    animation_grid,
    assemble_animation_frames,
    load_texture_metadata,
    rgb_with_transparency_bleed,
    split_animation_frames,
)


def test_split_animation_frames_handles_vertical_sheet() -> None:
    sheet = Image.new("RGB", (16, 48), "red")

    frames = split_animation_frames(sheet, {"animation": {"frametime": 2}})

    assert [frame.size for frame in frames] == [(16, 16), (16, 16), (16, 16)]


def test_split_animation_frames_ignores_non_animation_metadata() -> None:
    image = Image.new("RGB", (182, 5), "red")

    frames = split_animation_frames(image, {"gui": {"scaling": {"type": "stretch"}}})

    assert [frame.size for frame in frames] == [(182, 5)]


def test_animation_grid_uses_explicit_frame_size() -> None:
    sheet = Image.new("RGB", (32, 64))

    assert animation_grid(sheet, {"animation": {"width": 16, "height": 16}}) == (16, 16, 2, 4)


def test_assemble_animation_frames_round_trips_a_split_sheet() -> None:
    sheet = Image.new("RGB", (32, 32))
    for index, colour in enumerate(("red", "green", "blue", "white")):
        sheet.paste(Image.new("RGB", (16, 16), colour), ((index % 2) * 16, (index // 2) * 16))
    metadata = {"animation": {"width": 16, "height": 16}}

    frames = split_animation_frames(sheet, metadata)

    assert assemble_animation_frames(frames, 2, 2).tobytes() == sheet.tobytes()


def test_assemble_animation_frames_rejects_a_mismatched_grid() -> None:
    with pytest.raises(ValueError):
        assemble_animation_frames([Image.new("RGB", (4, 4))], 2, 2)


def test_load_texture_metadata_ignores_missing_and_invalid_files(tmp_path: Path) -> None:
    texture = tmp_path / "tex.png"
    assert load_texture_metadata(texture) == {}
    Path(f"{texture}.mcmeta").write_text("{not json")
    assert load_texture_metadata(texture) == {}
    Path(f"{texture}.mcmeta").write_text('{"animation": {}}')
    assert load_texture_metadata(texture) == {"animation": {}}


def test_rgb_with_transparency_bleed_fills_transparent_pixels_with_the_nearest_opaque_colour() -> None:
    source = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    source.putpixel((0, 0), (240, 80, 40, 255))

    bled = rgb_with_transparency_bleed(source)

    assert bled.mode == "RGB"
    assert bled.getpixel((1, 0)) == (240, 80, 40)
