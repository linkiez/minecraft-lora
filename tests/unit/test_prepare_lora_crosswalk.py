import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from prepare_lora_crosswalk import Entry, build_generation_rows, main, split_rows


def test_build_generation_rows_includes_paired_and_missing_textures(tmp_path: Path) -> None:
    paired = Entry("mods", tmp_path / "paired.png", "assets/mod/textures/paired.png")
    missing = Entry("original", tmp_path / "missing.png", "assets/minecraft/textures/missing.png")
    reference = Entry("resource_pack", tmp_path / "reference.png", paired.rel_key)

    rows = build_generation_rows(
        {paired.rel_key: paired, missing.rel_key: missing},
        {reference.rel_key: reference},
    )

    assert [row["rel_key"] for row in rows] == [missing.rel_key, paired.rel_key]
    assert rows[0]["resource_path"] == ""
    assert rows[1]["resource_path"] == str(reference.abs_path)


def write_texture(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def test_main_writes_pair_manifests_from_explicit_roots(tmp_path: Path) -> None:
    import csv
    import json

    write_texture(tmp_path / "original", "assets/minecraft/textures/block/stone.png")
    write_texture(tmp_path / "mods", "assets/mod/textures/block/gem.png")
    write_texture(tmp_path / "mods", "assets/mod/textures/block/ore.png")
    write_texture(tmp_path / "pack", "assets/mod/textures/block/gem.png")
    write_texture(tmp_path / "pack", "assets/other/textures/block/only_in_pack.png")
    out = tmp_path / "prep"

    exit_code = main(
        [
            "--original-assets", str(tmp_path / "original" / "assets"),
            "--mods-assets", str(tmp_path / "mods" / "assets"),
            "--resource-assets", str(tmp_path / "pack" / "assets"),
            "--output-dir", str(out),
        ]
    )

    assert exit_code == 0
    with (out / "pairs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["rel_key"] for row in rows] == ["assets/mod/textures/block/gem.png"]
    assert set(rows[0]) == {"rel_key", "base_source", "base_path", "resource_path"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["pairs_total"] == 1 and summary["base_only_total"] == 2 and summary["resourcepack_only_total"] == 1
    assert not list(out.glob("*upscaled*"))


def test_split_rows_uses_eighty_ten_ten_by_default() -> None:
    train, validation, test = split_rows([{"rel_key": str(index)} for index in range(100)])

    assert (len(train), len(validation), len(test)) == (80, 10, 10)
