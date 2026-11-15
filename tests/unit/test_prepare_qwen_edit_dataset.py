import csv
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from prepare_qwen_edit_training_dataset import prepare_dataset


def save_image(path: Path, size: tuple[int, int], color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def write_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_dataset_creates_matching_control_target_and_caption(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    save_image(source, (16, 32), "red")
    save_image(target, (64, 128), "blue")
    pairs = tmp_path / "pairs.csv"
    write_pairs(
        pairs,
        [{"rel_key": "assets/mod/textures/item/malachite.png", "base_path": str(source), "resource_path": str(target)}],
    )

    count, skipped = prepare_dataset(pairs, tmp_path / "dataset")

    output_target = tmp_path / "dataset/target/assets/mod/textures/item/malachite.png"
    output_control = tmp_path / "dataset/control/assets/mod/textures/item/malachite.png"
    assert (count, skipped) == (1, 0)
    assert output_target.is_file()
    assert output_control.is_file()
    assert output_target.with_suffix(".txt").read_text(encoding="utf-8").startswith("Redesign this Minecraft item")
    metadata = json.loads((tmp_path / "dataset/metadata.jsonl").read_text(encoding="utf-8"))
    assert Path(metadata["image_path"]).read_bytes() == output_target.read_bytes()
    assert Path(metadata["control_path"]).read_bytes() == output_control.read_bytes()
    assert metadata["caption"].startswith("Redesign this Minecraft item")


def test_prepare_dataset_expands_animation_frames_with_matching_names(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    save_image(source, (16, 32), "red")
    save_image(target, (16, 48), "blue")
    Path(f"{source}.mcmeta").write_text(json.dumps({"animation": {"frametime": 2}}), encoding="utf-8")
    Path(f"{target}.mcmeta").write_text(json.dumps({"animation": {"frametime": 2}}), encoding="utf-8")
    pairs = tmp_path / "pairs.csv"
    write_pairs(
        pairs,
        [{"rel_key": "assets/mod/textures/item/gem.png", "base_path": str(source), "resource_path": str(target)}],
    )

    count, skipped = prepare_dataset(pairs, tmp_path / "dataset")

    target_names = sorted(path.name for path in (tmp_path / "dataset/target/assets/mod/textures/item").glob("*.png"))
    control_names = sorted(path.name for path in (tmp_path / "dataset/control/assets/mod/textures/item").glob("*.png"))
    assert (count, skipped) == (3, 0)
    assert target_names == ["gem__frame_0000.png", "gem__frame_0001.png", "gem__frame_0002.png"]
    assert control_names == target_names
    assert len((tmp_path / "dataset/metadata.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_prepare_dataset_uses_unique_musubi_names_for_duplicate_basenames(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    save_image(source, (16, 16), "red")
    save_image(target, (32, 32), "blue")
    pairs = tmp_path / "pairs.csv"
    write_pairs(
        pairs,
        [
            {"rel_key": "assets/mod_a/textures/item/gem.png", "base_path": str(source), "resource_path": str(target)},
            {"rel_key": "assets/mod_b/textures/item/gem.png", "base_path": str(source), "resource_path": str(target)},
        ],
    )

    prepare_dataset(pairs, tmp_path / "dataset")

    records = [json.loads(line) for line in (tmp_path / "dataset/metadata.jsonl").read_text().splitlines()]
    image_names = [Path(record["image_path"]).name for record in records]
    control_names = [Path(record["control_path"]).name for record in records]
    assert len(set(image_names)) == 2
    assert image_names == control_names
    assert all(Path(record["image_path"]).parent.name == "musubi_target" for record in records)


def test_prepare_dataset_filters_category_and_uses_atlas_caption(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    save_image(source, (16, 16), "red")
    save_image(target, (32, 32), "blue")
    pairs = tmp_path / "pairs.csv"
    write_pairs(
        pairs,
        [
            {
                "rel_key": "assets/mod/textures/entity/citizen.png",
                "base_path": str(source),
                "resource_path": str(target),
            },
            {
                "rel_key": "assets/mod/textures/item/gem.png",
                "base_path": str(source),
                "resource_path": str(target),
            },
        ],
    )

    count, skipped = prepare_dataset(pairs, tmp_path / "dataset", {"entity"})

    entity_caption = tmp_path / "dataset/target/assets/mod/textures/entity/citizen.txt"
    assert (count, skipped) == (1, 1)
    assert entity_caption.is_file()
    assert "UV atlas" in entity_caption.read_text(encoding="utf-8")
    assert not (tmp_path / "dataset/target/assets/mod/textures/item/gem.png").exists()
