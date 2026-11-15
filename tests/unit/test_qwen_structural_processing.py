from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_structural_processing import load_source_image, vectorize_structural_frame
from qwen_texture_processing import process_texture


def test_process_texture_upscales_entity_with_antialiased_uv_alpha(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source_image = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
    source_image.paste(Image.new("RGBA", (3, 2), (180, 90, 40, 255)), (1, 1))
    source_image.save(source)
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/entity/citizen.png"
    render_calls = 0

    def render(_: Image.Image, __: str, ___: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        return Image.new("RGBA", (64, 64), (70, 160, 220, 255))

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=32,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    with Image.open(output) as generated:
        expected_alpha = source_image.getchannel("A").resize(generated.size, Image.Resampling.LANCZOS)
        assert generated.getchannel("A").tobytes() == expected_alpha.tobytes()
        assert any(0 < value < 255 for value in generated.getchannel("A").get_flattened_data())
    assert render_calls == 0
    assert output.with_name("citizen_n.png").is_file()
    assert output.with_name("citizen_s.png").is_file()


def test_process_texture_normalizes_entity_regions_to_vector_palette(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source_image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for x in range(3, 13):
        for y in range(3, 13):
            source_image.putpixel((x, y), ((x * 17) % 256, (y * 29) % 256, ((x + y) * 13) % 256, 255))
    source_image.save(source)
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/entity/citizen.png"

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=128,
        normal_strength=2.0,
        render_frame=lambda *_: Image.new("RGB", (64, 64)),
        overwrite=False,
    )
    with Image.open(output) as generated:
        pixels = generated.convert("RGBA").tobytes()
        visible = {
            (pixels[offset], pixels[offset + 1], pixels[offset + 2])
            for offset in range(0, len(pixels), 4)
            if pixels[offset + 3] > 127
        }
        assert len(visible) <= 32


def test_process_texture_upscales_gui_with_antialiased_alpha_and_no_pbr(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source_image = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
    source_image.paste(Image.new("RGBA", (3, 2), (210, 40, 70, 255)), (1, 1))
    source_image.save(source)
    Path(f"{source}.mcmeta").write_text('{"gui": {"scaling": {"type": "stretch"}}}')
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/gui/widgets.png"
    render_calls = 0

    def render(_: Image.Image, __: str, ___: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        return Image.new("RGB", (1024, 1024), (60, 180, 230))

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=128,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 0
    with Image.open(output) as generated:
        assert generated.size == (128, 64)
        assert generated.mode == "RGBA"
        expected_alpha = source_image.getchannel("A").resize(generated.size, Image.Resampling.LANCZOS)
        assert generated.getchannel("A").tobytes() == expected_alpha.tobytes()
        assert any(0 < value < 255 for value in generated.getchannel("A").get_flattened_data())
    assert output.read_bytes() != source.read_bytes()
    assert Path(f"{output}.mcmeta").read_bytes() == Path(f"{source}.mcmeta").read_bytes()
    assert not output.with_name("widgets_n.png").exists()
    assert not output.with_name("widgets_s.png").exists()

    assert not process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=128,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 0


def test_process_texture_copies_malformed_gui_without_rendering(tmp_path: Path) -> None:
    source = tmp_path / "broken.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\nmalformed")
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/gui/broken.png"
    render_calls = 0

    def render(_: Image.Image, __: str, ___: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        return Image.new("RGB", (16, 16), (255, 255, 255))

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=4096,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 0
    assert output.read_bytes() == source.read_bytes()

    assert not process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=4096,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 0


def test_load_source_image_expands_palette_transparency_to_alpha(tmp_path: Path) -> None:
    path = tmp_path / "palette.png"
    palette = Image.new("P", (4, 4), 0)
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * 250 * 3)
    palette.putpixel((1, 1), 1)
    palette.save(path, transparency=0)

    loaded = load_source_image(path)

    assert loaded.mode == "RGBA"
    assert loaded.getpixel((0, 0))[3] == 0
    assert loaded.getpixel((1, 1)) == (0, 255, 0, 255)


def test_vectorize_structural_frame_honours_palette_colors() -> None:
    frame = Image.new("RGB", (16, 16))
    for x in range(16):
        for y in range(16):
            frame.putpixel((x, y), (x * 16, y * 16, (x + y) * 8))

    reduced = vectorize_structural_frame(frame, 64, palette_colors=4)
    wider = vectorize_structural_frame(frame, 64, palette_colors=16)

    assert len(reduced.getcolors(maxcolors=4096)) <= 4
    assert len(wider.getcolors(maxcolors=4096)) > 4
