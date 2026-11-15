import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from extract_jar_textures import extract_jar_textures


def write_jar(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_extracts_pngs_and_animation_metadata_without_the_assets_prefix(tmp_path: Path) -> None:
    jars = tmp_path / "jars"
    jars.mkdir()
    write_jar(
        jars / "mod.jar",
        {
            "assets/mod/textures/block/a.png": b"png",
            "assets/mod/textures/block/a.png.mcmeta": b"{}",
            "assets/mod/lang/en_us.json": b"{}",
            "data/mod/recipes/x.json": b"{}",
        },
    )
    output = tmp_path / "out"

    result = extract_jar_textures(jars, output)

    assert (output / "mod/textures/block/a.png").read_bytes() == b"png"
    assert (output / "mod/textures/block/a.png.mcmeta").is_file()
    assert not (output / "mod/lang").exists()
    assert result.extracted == 1 and result.errors == 0


def test_skips_jars_without_textures_and_invalid_archives(tmp_path: Path) -> None:
    jars = tmp_path / "jars"
    jars.mkdir()
    write_jar(jars / "empty.jar", {"data/x.json": b"{}"})
    (jars / "broken.jar").write_bytes(b"not a zip")

    result = extract_jar_textures(jars, tmp_path / "out")

    assert result.extracted == 0 and result.skipped == 2 and result.errors == 0


def test_does_not_overwrite_existing_files_or_escape_the_output_directory(tmp_path: Path) -> None:
    jars = tmp_path / "jars"
    jars.mkdir()
    write_jar(
        jars / "mod.jar",
        {
            "assets/mod/textures/a.png": b"new",
            "assets/../../escape.png": b"evil",
        },
    )
    output = tmp_path / "out"
    (output / "mod/textures").mkdir(parents=True)
    (output / "mod/textures/a.png").write_bytes(b"old")

    extract_jar_textures(jars, output)

    assert (output / "mod/textures/a.png").read_bytes() == b"old"
    assert not (tmp_path / "escape.png").exists()
