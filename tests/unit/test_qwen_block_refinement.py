import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_texture_processing import finish_creative_block_quality, refine_block_definition


def test_refine_block_definition_preserves_source_internal_structure() -> None:
    source = Image.new("RGB", (8, 8), (40, 40, 40))
    for x in range(4, 8):
        for y in range(8):
            source.putpixel((x, y), (220, 220, 220))

    generated = Image.new("RGB", (64, 64), (150, 90, 90))
    refined = refine_block_definition(generated, source, resolution=64, animated=False).convert("RGB")

    left_half = refined.crop((0, 0, 32, 64)).convert("L")
    right_half = refined.crop((32, 0, 64, 64)).convert("L")
    assert sum(left_half.tobytes()) != sum(right_half.tobytes())
    assert sum(right_half.tobytes()) > sum(left_half.tobytes())


def test_refine_block_definition_reuses_source_color_layout() -> None:
    source = Image.new("RGB", (8, 8), (30, 60, 180))
    for x in range(4, 8):
        for y in range(8):
            source.putpixel((x, y), (220, 200, 70))

    generated = Image.new("RGB", (64, 64), (150, 90, 90))
    refined = refine_block_definition(generated, source, resolution=64, animated=False).convert("RGB")

    left_half = refined.crop((0, 0, 32, 64))
    right_half = refined.crop((32, 0, 64, 64))
    left_mean = ImageStat.Stat(left_half).mean
    right_mean = ImageStat.Stat(right_half).mean

    assert left_mean[2] > right_mean[2]
    assert right_mean[0] > left_mean[0]
    assert right_mean[1] > left_mean[1]


def test_finish_creative_block_quality_increases_edge_contrast() -> None:
    blurred = Image.new("RGB", (64, 64), (100, 100, 100))
    for x in range(32, 64):
        for y in range(64):
            blurred.putpixel((x, y), (140, 140, 140))
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=3.0))

    finished = finish_creative_block_quality(blurred, animated=False).convert("L")

    assert finished.getpixel((34, 32)) - finished.getpixel((29, 32)) > 20


def test_refine_block_definition_does_not_imprint_source_pixel_grid() -> None:
    source = Image.new("RGB", (8, 8))
    for x in range(8):
        for y in range(8):
            source.putpixel((x, y), (200, 180, 120) if (x + y) % 2 else (90, 70, 40))
    generated = Image.new("RGB", (256, 256), (150, 120, 80))

    refined = refine_block_definition(generated, source, resolution=256, animated=False).convert("L")

    steps = ImageChops.difference(refined, ImageChops.offset(refined, 1, 0)).crop((1, 0, 256, 256))
    assert steps.getextrema()[1] <= 3


def test_refine_block_definition_keeps_grayscale_sources_tintable() -> None:
    source = Image.new("RGBA", (8, 8), (180, 180, 180, 200))
    generated = Image.new("RGB", (64, 64), (60, 160, 40))

    refined = refine_block_definition(generated, source, resolution=64, animated=True).convert("RGB")

    red, green, blue = refined.getpixel((32, 32))
    assert red == green == blue


def test_refine_block_definition_pulls_layout_to_source_but_keeps_generated_detail() -> None:
    source = Image.new("RGB", (8, 8), (40, 40, 200))
    source.paste(Image.new("RGB", (4, 8), (200, 40, 40)), (4, 0))
    generated = Image.new("RGB", (128, 128), (120, 120, 120))
    for x in range(0, 128, 8):
        generated.paste(Image.new("RGB", (4, 128), (170, 170, 170)), (x, 0))

    refined = refine_block_definition(generated, source, resolution=128, animated=False).convert("RGB")

    left, right = refined.getpixel((16, 64)), refined.getpixel((112, 64))
    assert left[2] > left[0] and right[0] > right[2]
    stripe, gap = refined.getpixel((2, 64)), refined.getpixel((6, 64))
    assert sum(stripe) - sum(gap) > 60
