import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_cutout_keying import (
    KEY_COLOURS,
    background_clause,
    choose_key_colour,
    extract_cutout,
    flatten_on_key,
    is_binary_cutout,
    key_alpha,
)

OBJECT_COLOUR = (200, 40, 40)


def cutout_source(size: int = 8, object_box: tuple[int, int, int, int] = (2, 2, 6, 6)) -> Image.Image:
    """Opaque red square on a fully transparent background."""
    frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    left, top, right, bottom = object_box
    frame.paste(Image.new("RGBA", (right - left, bottom - top), (*OBJECT_COLOUR, 255)), (left, top))
    return frame


def rendered_on_key(key: tuple[int, int, int], size: int, box: tuple[int, int, int, int]) -> Image.Image:
    image = Image.new("RGB", (size, size), key)
    left, top, right, bottom = box
    image.paste(Image.new("RGB", (right - left, bottom - top), OBJECT_COLOUR), (left, top))
    return image


def test_binary_cutout_requires_transparent_and_opaque_pixels_without_partial_alpha() -> None:
    assert is_binary_cutout(cutout_source())
    assert not is_binary_cutout(Image.new("RGBA", (8, 8), (10, 20, 30, 255)))
    assert not is_binary_cutout(Image.new("RGBA", (8, 8), (10, 20, 30, 0)))
    assert not is_binary_cutout(Image.new("RGB", (8, 8), (10, 20, 30)))
    translucent = cutout_source()
    translucent.putpixel((3, 3), (200, 40, 40, 120))
    assert not is_binary_cutout(translucent)


def test_key_colour_avoids_colours_present_in_the_object() -> None:
    magenta_object = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    magenta_object.paste(Image.new("RGBA", (4, 4), (250, 10, 250, 255)), (2, 2))
    cyan_object = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    cyan_object.paste(Image.new("RGBA", (4, 4), (10, 240, 240, 255)), (2, 2))

    assert choose_key_colour(magenta_object).name != "magenta"
    assert choose_key_colour(cyan_object).name != "cyan"
    assert choose_key_colour(cutout_source()).rgb in KEY_COLOURS.values()


def test_flatten_on_key_fills_transparency_with_the_key_colour_only() -> None:
    key = choose_key_colour(cutout_source())
    flat = flatten_on_key(cutout_source(), key)

    assert flat.mode == "RGB"
    assert flat.getpixel((0, 0)) == key.rgb
    assert flat.getpixel((3, 3)) == OBJECT_COLOUR


def test_background_clause_names_the_colour_and_forbids_it_in_the_object() -> None:
    key = KEY_COLOURS["magenta"]
    clause = background_clause(choose_key_colour(Image.new("RGBA", (2, 2), (0, 0, 0, 0))))

    assert "#" in clause and "solid" in clause and "background" in clause
    assert key in KEY_COLOURS.values()


def test_key_alpha_is_zero_on_the_key_one_on_the_object_and_partial_between() -> None:
    key = KEY_COLOURS["magenta"]
    image = Image.new("RGB", (3, 1))
    image.putpixel((0, 0), key)
    image.putpixel((1, 0), OBJECT_COLOUR)
    half = tuple((channel_key + channel_object) // 2 for channel_key, channel_object in zip(key, OBJECT_COLOUR, strict=True))
    image.putpixel((2, 0), half)

    alpha = key_alpha(image, key)

    assert alpha[0, 0] == 0.0 and alpha[0, 1] == 1.0
    assert 0.2 < alpha[0, 2] < 0.95


def test_extract_cutout_edge_pixels_take_the_object_colour_not_a_key_mix() -> None:
    key = KEY_COLOURS["magenta"]
    source = cutout_source(8, (0, 0, 5, 8))
    rendered = rendered_on_key(key, 64, (0, 0, 40, 64))
    half = tuple((channel_key + channel_object) // 2 for channel_key, channel_object in zip(key, OBJECT_COLOUR, strict=True))
    rendered.paste(Image.new("RGB", (1, 64), half), (40, 0))

    extracted = extract_cutout(rendered, source, key, (64, 64))

    assert extracted is not None
    rgb, alpha = extracted
    assert 0 < alpha.getpixel((40, 32)) < 255
    assert rgb.getpixel((40, 32)) == OBJECT_COLOUR


def test_extract_cutout_follows_the_drawn_contour_inside_the_source_silhouette() -> None:
    key = KEY_COLOURS["cyan"]
    source = cutout_source(8, (1, 1, 7, 7))
    # The model draws a smaller object than the source silhouette: its own contour must win at the edges.
    rendered = rendered_on_key(key, 64, (12, 12, 52, 52))

    extracted = extract_cutout(rendered, source, key, (64, 64))

    assert extracted is not None
    _, alpha = extracted
    assert alpha.getpixel((32, 32)) == 255
    assert alpha.getpixel((2, 2)) == 0
    assert alpha.getpixel((10, 32)) == 0


def test_extract_cutout_discards_content_drawn_far_outside_the_source_silhouette() -> None:
    key = KEY_COLOURS["cyan"]
    source = cutout_source(8, (0, 0, 3, 3))
    rendered = rendered_on_key(key, 64, (0, 0, 24, 24))
    rendered.paste(Image.new("RGB", (16, 16), OBJECT_COLOUR), (44, 44))

    extracted = extract_cutout(rendered, source, key, (64, 64))

    assert extracted is not None
    _, alpha = extracted
    assert alpha.getpixel((12, 12)) == 255
    assert alpha.getpixel((52, 52)) == 0


def test_extract_cutout_falls_back_when_the_model_filled_the_object_with_background() -> None:
    key = KEY_COLOURS["cyan"]
    rendered = Image.new("RGB", (64, 64), key)

    assert extract_cutout(rendered, cutout_source(), key, (64, 64)) is None


def test_extract_cutout_fills_enclosed_key_specks_inside_the_source_silhouette_without_leaking_the_key() -> None:
    key = KEY_COLOURS["cyan"]
    source = cutout_source(8, (0, 0, 8, 6))
    rendered = rendered_on_key(key, 64, (0, 0, 64, 48))
    rendered.paste(Image.new("RGB", (8, 8), key), (28, 20))  # model painted a key-coloured speck inside

    extracted = extract_cutout(rendered, source, key, (64, 64))

    assert extracted is not None
    rgb, alpha = extracted
    assert alpha.getpixel((32, 24)) == 255
    leaked = np.linalg.norm(np.array(rgb.getpixel((32, 24)), dtype=float) - np.array(key, dtype=float))
    assert leaked > 150


def test_extract_cutout_keeps_holes_that_the_source_also_has() -> None:
    key = KEY_COLOURS["cyan"]
    source = cutout_source(8, (0, 0, 8, 8))
    source.paste(Image.new("RGBA", (2, 2), (0, 0, 0, 0)), (3, 3))  # a window through the block
    rendered = rendered_on_key(key, 64, (0, 0, 64, 64))
    rendered.paste(Image.new("RGB", (16, 16), key), (24, 24))

    extracted = extract_cutout(rendered, source, key, (64, 64))

    assert extracted is not None
    assert extracted[1].getpixel((32, 32)) == 0


def test_extract_cutout_does_not_pad_the_drawing_out_to_the_source_silhouette() -> None:
    key = KEY_COLOURS["cyan"]
    source = cutout_source(8, (0, 0, 8, 8))
    rendered = rendered_on_key(key, 64, (8, 8, 56, 56))

    extracted = extract_cutout(rendered, source, key, (64, 64))

    assert extracted is not None
    assert extracted[1].getpixel((3, 3)) == 0


@pytest.mark.parametrize("name", sorted(KEY_COLOURS))
def test_every_key_colour_is_fully_saturated(name: str) -> None:
    assert max(KEY_COLOURS[name]) == 255 and min(KEY_COLOURS[name]) == 0
