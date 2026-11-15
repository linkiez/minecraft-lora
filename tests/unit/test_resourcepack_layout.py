import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from resourcepack_layout import copy_pack_icon, copy_texture_metadata, discover_textures, target_size, write_pack_metadata


def test_target_size_preserves_portrait_aspect_ratio() -> None:
    assert target_size((16, 32), 512) == (256, 512)


def test_target_size_preserves_landscape_aspect_ratio() -> None:
    assert target_size((32, 16), 512) == (512, 256)


def test_discover_textures_lets_mods_override_vanilla_and_skips_pbr_maps(tmp_path: Path) -> None:
    for root, name in (("original", "stone.png"), ("mods", "stone.png"), ("mods", "stone_n.png")):
        path = tmp_path / root / "assets/minecraft/textures/block" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2)).save(path)

    textures = discover_textures(tmp_path)

    assert list(textures) == [Path("assets/minecraft/textures/block/stone.png")]
    assert "mods" in textures[Path("assets/minecraft/textures/block/stone.png")].parts


def test_write_pack_metadata_records_the_resolution(tmp_path: Path) -> None:
    write_pack_metadata(tmp_path, 1024)

    metadata = json.loads((tmp_path / "pack.mcmeta").read_text())
    assert "1024px" in metadata["pack"]["description"]


def test_copy_texture_metadata_and_pack_icon_copy_only_existing_files(tmp_path: Path) -> None:
    source = tmp_path / "a.png"
    output = tmp_path / "out" / "a.png"
    output.parent.mkdir()
    copy_texture_metadata(source, output)
    assert not Path(f"{output}.mcmeta").exists()
    Path(f"{source}.mcmeta").write_text("{}")
    copy_texture_metadata(source, output)
    assert Path(f"{output}.mcmeta").is_file()

    copy_pack_icon(tmp_path, output.parent)
    assert not (output.parent / "pack.png").exists()
    Image.new("RGB", (4, 4)).save(tmp_path / "pack.png")
    copy_pack_icon(tmp_path, output.parent)
    assert (output.parent / "pack.png").is_file()
