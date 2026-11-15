import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_texture_processing import (
    process_texture,
    seed_key_for,
)


def test_process_texture_always_uses_source_as_block_conditioning_input(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (8, 8), (20, 30, 40, 255)).save(source)
    relative_path = Path("assets/mod/textures/block/sample.png")
    output_root = tmp_path / "pack"
    output = output_root / relative_path
    seen_pixels: list[tuple[int, int, int, int]] = []

    def render(frame: Image.Image, _: str, __: str, ___: str) -> Image.Image:
        seen_pixels.append(frame.convert("RGBA").getpixel((0, 0)))
        return Image.new("RGB", (64, 64), (100, 100, 100))

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=64,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )

    assert seen_pixels == [(20, 30, 40, 255)]


def test_process_texture_reassembles_animation_generates_pbr_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    sheet = Image.new("RGBA", (16, 32), (0, 0, 0, 0))
    sheet.paste(Image.new("RGBA", (16, 16), (20, 180, 80, 255)), (0, 0))
    sheet.paste(Image.new("RGBA", (16, 16), (30, 120, 220, 255)), (0, 16))
    sheet.save(source)
    Path(f"{source}.mcmeta").write_text(json.dumps({"animation": {"frametime": 2}}))
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/item/gem.png"
    render_calls = 0

    def render(frame: Image.Image, _: str, frame_key: str, __: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        color = (40, 210, 90) if frame_key.endswith("0000") else (60, 130, 240)
        return Image.new("RGB", (1024, 1024), color)

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=256,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 2
    with Image.open(output) as generated:
        assert generated.size == (256, 512)
        assert generated.mode == "RGBA"
    assert Path(f"{output}.mcmeta").is_file()
    assert output.with_name("gem_n.png").is_file()
    assert output.with_name("gem_s.png").is_file()
    assert Path(f"{output.with_name('gem_n.png')}.mcmeta").is_file()

    assert not process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=256,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 2


def test_process_texture_replaces_corrupt_resume_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (16, 16), (20, 180, 80, 255)).save(source)
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/item/gem.png"
    output.parent.mkdir(parents=True)
    output.touch()
    render_calls = 0

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        return Image.new("RGB", (1024, 1024), (40, 210, 90))

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=256,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 1
    with Image.open(output) as generated:
        generated.verify()

    assert not process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=256,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )
    assert render_calls == 1


def test_process_texture_saves_opaque_block_without_transparency(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (16, 16), (110, 80, 40, 255)).save(source)
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/block/oak_planks.png"

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=128,
        normal_strength=2.0,
        render_frame=lambda *_: Image.new("RGBA", (64, 64), (180, 140, 90, 255)),
        overwrite=False,
    )

    with Image.open(output) as generated:
        assert generated.mode == "RGBA"
        assert generated.getchannel("A").getextrema() == (255, 255)


def test_process_texture_renders_opaque_animated_block_with_anchored_creative_consistency(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    sheet = Image.new("RGB", (8, 48), (0, 0, 0))
    for index, color in enumerate(((140, 40, 60), (150, 50, 70), (160, 60, 80), (170, 70, 90), (180, 80, 100), (190, 90, 110))):
        frame = Image.new("RGB", (8, 8), color)
        sheet.paste(frame, (0, index * 8))
    sheet.save(source)
    Path(f"{source}.mcmeta").write_text('{"animation": {"interpolate": true, "frametime": 4}}')
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/block/beetroot_soup_cauldron.png"
    render_calls = 0

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        return Image.new("RGB", (64, 64), (140, 70, 90))

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=64,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )

    assert render_calls == 1
    with Image.open(output) as generated:
        assert generated.size == (64, 384)
        assert generated.mode == "RGBA"
        assert generated.getchannel("A").getextrema() == (255, 255)
        first_frame = generated.crop((0, 0, 64, 64))
        last_frame = generated.crop((0, 320, 64, 384))
        assert first_frame.tobytes() != last_frame.tobytes()
    assert Path(f"{output}.mcmeta").read_text() == Path(f"{source}.mcmeta").read_text()
    assert output.with_name("beetroot_soup_cauldron_n.png").is_file()
    assert output.with_name("beetroot_soup_cauldron_s.png").is_file()


def test_process_texture_renders_animated_block_with_non_opaque_source_alpha_using_anchored_creative_flow(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    sheet = Image.new("LA", (8, 16), (120, 212))
    sheet.paste(Image.new("LA", (8, 8), (140, 212)), (0, 8))
    sheet.save(source)
    Path(f"{source}.mcmeta").write_text('{"animation": {"frametime": 4}}')
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/block/dye_flow.png"
    render_calls = 0

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        return Image.new("RGB", (64, 64), (100, 130, 180))

    assert process_texture(
        source,
        output,
        output_root,
        tmp_path / "references",
        resolution=64,
        normal_strength=2.0,
        render_frame=render,
        overwrite=False,
    )

    assert render_calls == 1
    with Image.open(output) as generated:
        assert generated.size == (64, 128)
        assert generated.mode == "RGBA"
        assert generated.getchannel("A").getextrema() == (212, 212)
        first_frame = generated.crop((0, 0, 64, 64))
        last_frame = generated.crop((0, 64, 64, 128))
        assert first_frame.tobytes() != last_frame.tobytes()








def test_process_texture_resume_regenerates_missing_pbr_maps(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (16, 16), (20, 180, 80, 255)).save(source)
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/item/gem.png"
    render_calls = 0

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        nonlocal render_calls
        render_calls += 1
        return Image.new("RGB", (1024, 1024), (40, 210, 90))

    args = (source, output, output_root, tmp_path / "references")
    kwargs = {"resolution": 64, "normal_strength": 2.0, "render_frame": render, "overwrite": False}
    assert process_texture(*args, **kwargs)
    output.with_name("gem_n.png").unlink()
    output.with_name("gem_s.png").unlink()

    assert not process_texture(*args, **kwargs)

    assert render_calls == 1
    assert output.with_name("gem_n.png").is_file()
    assert output.with_name("gem_s.png").is_file()


def test_process_texture_keeps_previous_output_when_overwrite_render_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (16, 16), (20, 180, 80, 255)).save(source)
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/item/gem.png"

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        return Image.new("RGB", (1024, 1024), (40, 210, 90))

    def failing_render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        raise RuntimeError("ComfyUI timed out")

    common = {"resolution": 64, "normal_strength": 2.0, "overwrite": True}
    assert process_texture(source, output, output_root, tmp_path / "refs", render_frame=render, **common)
    previous = output.read_bytes()

    with pytest.raises(RuntimeError):
        process_texture(source, output, output_root, tmp_path / "refs", render_frame=failing_render, **common)

    assert output.read_bytes() == previous


def test_anchored_animated_block_refines_every_frame_exactly_once(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    sheet = Image.new("RGBA", (16, 48), (90, 140, 70, 255))
    for offset in (0, 16, 32):
        sheet.paste(Image.new("RGBA", (8, 8), (30, 60, 20, 255)), (4, offset + 4))
    sheet.save(source)
    Path(f"{source}.mcmeta").write_text(json.dumps({"animation": {"frametime": 2}}))
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/block/moss.png"

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        return Image.new("RGB", (1024, 1024), (80, 150, 60))

    process_texture(
        source, output, output_root, tmp_path / "refs",
        resolution=64, normal_strength=2.0, render_frame=render, overwrite=False,
    )

    with Image.open(output) as generated:
        frames = [generated.crop((0, row * 64, 64, (row + 1) * 64)).convert("RGB") for row in range(3)]
    assert ImageChops.difference(frames[0], frames[1]).getbbox() is None
    assert ImageChops.difference(frames[0], frames[2]).getbbox() is None




def test_animated_block_frames_share_anchor_detail_and_follow_source_motion(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    sheet = Image.new("RGBA", (8, 16), (120, 60, 160, 200))
    sheet.paste(Image.new("RGBA", (4, 8), (220, 160, 240, 200)), (0, 0))
    sheet.paste(Image.new("RGBA", (4, 8), (220, 160, 240, 200)), (4, 8))
    sheet.save(source)
    Path(f"{source}.mcmeta").write_text('{"animation": {"frametime": 4}}')
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/block/potion.png"
    seeds: list[int] = []

    def render(_: Image.Image, __: str, frame_key: str, ___: str) -> Image.Image:
        seeds.append(len(seeds))
        # A different colour per call: independent per-frame renders would be visibly inconsistent.
        return Image.new("RGB", (64, 64), (150, 90, 170) if len(seeds) == 1 else (40, 200, 40))

    process_texture(
        source, output, output_root, tmp_path / "refs",
        resolution=64, normal_strength=2.0, render_frame=render, overwrite=False,
    )

    assert len(seeds) == 1
    with Image.open(output) as generated:
        first = generated.crop((0, 0, 64, 64)).convert("RGB")
        second = generated.crop((0, 64, 64, 128)).convert("RGB")
        assert generated.getchannel("A").getextrema() == (200, 200)
    # The bright half moves from left (frame 0) to right (frame 1), as in the source.
    assert ImageStat.Stat(first.crop((0, 0, 24, 64))).mean[0] > ImageStat.Stat(first.crop((40, 0, 64, 64))).mean[0]
    assert ImageStat.Stat(second.crop((40, 0, 64, 64))).mean[0] > ImageStat.Stat(second.crop((0, 0, 24, 64))).mean[0]
    # Frame 1 keeps the anchor palette instead of the colours of an independent second render.
    second_mean = ImageStat.Stat(second).mean
    assert second_mean[1] < second_mean[0] and second_mean[1] < second_mean[2]




def test_scrolling_animation_moves_the_anchor_drawing_with_the_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    frame = Image.new("RGB", (16, 16), (40, 60, 200))
    frame.paste(Image.new("RGB", (16, 4), (200, 220, 255)), (0, 0))
    sheet = Image.new("RGB", (16, 64))
    for index in range(4):
        sheet.paste(ImageChops.offset(frame, 0, index * 4), (0, index * 16))
    sheet.save(source)
    Path(f"{source}.mcmeta").write_text('{"animation": {}}')
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/block/water_flow.png"

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        drawing = Image.new("RGB", (64, 64), (30, 50, 180))
        drawing.paste(Image.new("RGB", (8, 8), (255, 255, 0)), (28, 4))
        return drawing

    process_texture(
        source, output, output_root, tmp_path / "refs",
        resolution=64, normal_strength=2.0, render_frame=render, overwrite=False,
    )

    with Image.open(output) as generated:
        first = generated.crop((0, 0, 64, 64)).convert("RGB")
        third = generated.crop((0, 128, 64, 192)).convert("RGB")
    # The yellow mark drawn by the model travels 2 frames * 4 source px * 4x scale = 32 px down.
    def yellowness(image: Image.Image, point: tuple[int, int]) -> int:
        red, _, blue = image.getpixel(point)
        return red - blue

    assert yellowness(first, (32, 8)) > 0 > yellowness(first, (32, 40))
    assert yellowness(third, (32, 40)) > 0 > yellowness(third, (32, 8))


def test_translucent_block_takes_its_silhouette_from_the_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    sheet = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    sheet.paste(Image.new("RGBA", (4, 8), (60, 140, 50, 250)), (4, 0))
    sheet.save(source)
    output_root = tmp_path / "pack"
    output = output_root / "assets/mod/textures/block/fern.png"

    def render(_: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        return Image.new("RGB", (64, 64), (70, 160, 60))

    process_texture(
        source, output, output_root, tmp_path / "refs",
        resolution=64, normal_strength=2.0, render_frame=render, overwrite=False,
    )

    with Image.open(output) as generated:
        alpha = generated.getchannel("A")
    assert alpha.getpixel((8, 32)) == 0
    assert alpha.getpixel((56, 32)) == 250
    assert alpha.getpixel((28, 32)) < 40 and alpha.getpixel((36, 32)) > 200


def test_animated_non_block_frames_share_one_seed() -> None:
    path = Path("assets/mod/textures/item/compass.png")

    assert seed_key_for(path, 0, 8) == seed_key_for(path, 5, 8)
    assert seed_key_for(path, 0, 1) != seed_key_for(Path("assets/mod/textures/item/clock.png"), 0, 1)


def test_translucent_block_is_conditioned_on_material_colour_instead_of_black(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    sheet = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    sheet.paste(Image.new("RGBA", (4, 8), (60, 140, 50, 250)), (4, 0))
    sheet.save(source)
    output_root = tmp_path / "pack"
    seen: list[Image.Image] = []

    def render(frame: Image.Image, __: str, ___: str, ____: str) -> Image.Image:
        seen.append(frame)
        return Image.new("RGB", (64, 64), (70, 160, 60))

    process_texture(
        source, output_root / "assets/mod/textures/block/fern.png", output_root, tmp_path / "refs",
        resolution=64, normal_strength=2.0, render_frame=render, overwrite=False,
    )

    assert seen[0].convert("RGBA").getpixel((0, 0)) == (60, 140, 50, 255)
    assert seen[0].convert("RGBA").getpixel((6, 4)) == (60, 140, 50, 255)


