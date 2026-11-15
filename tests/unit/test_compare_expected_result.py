from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from compare_expected_result import compare_expected_to_result, resolve_expected_path, resolve_result_path


def test_compare_expected_to_result_scores_identical_images_higher(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    expected = tmp_path / "expected.png"
    identical = tmp_path / "identical.png"
    different = tmp_path / "different.png"

    base = Image.new("RGBA", (8, 8), (40, 80, 160, 255))
    for x in range(4, 8):
        for y in range(8):
            base.putpixel((x, y), (200, 180, 60, 255))
    base.save(expected)
    base.save(identical)
    Image.new("RGBA", (8, 8), (160, 80, 90, 255)).save(different)

    identical_metrics = compare_expected_to_result(expected, identical)
    different_metrics = compare_expected_to_result(expected, different)

    assert identical_metrics["composite_score"] == 1.0
    assert identical_metrics["edge_similarity"] >= different_metrics["edge_similarity"]
    assert identical_metrics["luminance_similarity"] > different_metrics["luminance_similarity"]
    assert identical_metrics["color_similarity"] > different_metrics["color_similarity"]


def test_compare_path_resolution_defaults_to_texture_and_output_roots() -> None:
    relative = Path("assets/example/textures/block/sample.png")

    assert resolve_expected_path(Path("texture"), relative) == Path("texture") / relative
    assert resolve_result_path(Path("output/test_pack"), relative) == Path("output/test_pack") / relative
