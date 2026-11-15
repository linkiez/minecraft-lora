import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_cutout_keying import KEY_COLOURS
from qwen_texture_processing import process_texture

RELATIVE_PATH = Path("assets/mod/textures/block/sprout.png")
OBJECT_COLOUR = (60, 160, 40)


def write_cutout_source(path: Path, alpha_value: int = 255) -> None:
    frame = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    frame.paste(Image.new("RGBA", (8, 12), (*OBJECT_COLOUR, alpha_value)), (4, 2))
    frame.save(path)


def run_process(tmp_path: Path, render) -> Image.Image:
    source = tmp_path / "source.png"
    output_root = tmp_path / "pack"
    process_texture(
        source, output_root / RELATIVE_PATH, output_root, tmp_path / "refs",
        resolution=64, normal_strength=2.0, render_frame=render, overwrite=False,
    )
    return Image.open(output_root / RELATIVE_PATH).convert("RGBA")


def keyed_render(frame: Image.Image) -> Image.Image:
    """Stand-in model: redraws a slightly different shape on the same flat background it was given."""
    background = frame.convert("RGB").getpixel((0, 0))
    drawing = Image.new("RGB", (64, 64), background)
    drawing.paste(Image.new("RGB", (28, 44), (20, 200, 60)), (18, 8))
    return drawing


def test_cutout_block_is_rendered_on_a_key_colour_and_keyed_out_by_script(tmp_path: Path) -> None:
    write_cutout_source(tmp_path / "source.png")
    seen: list[tuple[tuple[int, int, int], str]] = []

    def render(frame: Image.Image, prompt: str, _: str, __: str) -> Image.Image:
        seen.append((frame.convert("RGB").getpixel((0, 0)), prompt))
        return keyed_render(frame)

    result = run_process(tmp_path, render)

    assert len(seen) == 1
    corner_colour, prompt = seen[0]
    assert corner_colour in KEY_COLOURS.values()
    assert "flat solid" in prompt and "background" in prompt
    alpha = np.asarray(result.getchannel("A"))
    assert alpha[32, 32] == 255 and alpha[2, 2] == 0 and alpha[32, 60] == 0
    visible = np.asarray(result)[alpha > 0][:, :3].astype(float)
    key = np.array(corner_colour, dtype=float)
    assert np.linalg.norm(visible - key, axis=1).min() > 100


def test_translucent_block_keeps_the_source_alpha_path_without_a_key_background(tmp_path: Path) -> None:
    write_cutout_source(tmp_path / "source.png", alpha_value=120)
    seen: list[tuple[tuple[int, int, int], str]] = []

    def render(frame: Image.Image, prompt: str, _: str, __: str) -> Image.Image:
        seen.append((frame.convert("RGB").getpixel((0, 0)), prompt))
        return Image.new("RGB", (64, 64), (20, 200, 60))

    result = run_process(tmp_path, render)

    assert seen[0][0] == OBJECT_COLOUR
    assert "flat solid" not in seen[0][1]
    assert result.getpixel((32, 32))[3] == 120


def test_failed_keying_falls_back_to_the_source_alpha_with_a_second_render(tmp_path: Path) -> None:
    write_cutout_source(tmp_path / "source.png")
    seen: list[tuple[int, int, int]] = []

    def render(frame: Image.Image, _: str, __: str, ___: str) -> Image.Image:
        seen.append(frame.convert("RGB").getpixel((0, 0)))
        if len(seen) == 1:
            return Image.new("RGB", (64, 64), frame.convert("RGB").getpixel((0, 0)))
        return Image.new("RGB", (64, 64), (20, 200, 60))

    result = run_process(tmp_path, render)

    assert len(seen) == 2 and seen[1] == OBJECT_COLOUR
    assert result.getpixel((32, 32))[3] == 255 and result.getpixel((2, 2))[3] == 0


def test_animated_cutout_block_keeps_the_per_frame_source_alpha_path(tmp_path: Path) -> None:
    sheet = Image.new("RGBA", (16, 32), (0, 0, 0, 0))
    for index in range(2):
        sheet.paste(Image.new("RGBA", (8, 12), (*OBJECT_COLOUR, 255)), (4, index * 16 + 2))
    sheet.save(tmp_path / "source.png")
    Path(f"{tmp_path / 'source.png'}.mcmeta").write_text('{"animation": {}}')
    seen: list[str] = []

    def render(frame: Image.Image, prompt: str, _: str, __: str) -> Image.Image:
        seen.append(prompt)
        return Image.new("RGB", (64, 64), (20, 200, 60))

    run_process(tmp_path, render)

    assert len(seen) == 1 and "flat solid" not in seen[0]
