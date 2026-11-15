import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_resourcepack_batch import compare_generated_texture_against_expected


def test_compare_generated_texture_against_expected_returns_metrics_for_matching_pair(tmp_path: Path) -> None:
    relative_path = Path("assets/example/textures/block/sample.png")
    output_root = tmp_path / "output"
    expected_root = tmp_path / "expected"
    expected_path = expected_root / relative_path
    result_path = output_root / relative_path
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    base = Image.new("RGBA", (8, 8), (30, 50, 160, 255))
    for x in range(4, 8):
        for y in range(8):
            base.putpixel((x, y), (220, 200, 70, 255))
    base.save(expected_path)
    base.save(result_path)

    metrics = compare_generated_texture_against_expected(relative_path, output_root, expected_root, 0.8, 0.8)

    assert metrics is not None
    assert metrics["edge_similarity"] == 1.0
    assert metrics["composite_score"] == 1.0


def test_compare_generated_texture_against_expected_raises_when_threshold_is_not_met(tmp_path: Path) -> None:
    relative_path = Path("assets/example/textures/block/sample.png")
    output_root = tmp_path / "output"
    expected_root = tmp_path / "expected"
    expected_path = expected_root / relative_path
    result_path = output_root / relative_path
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    base = Image.new("RGBA", (8, 8), (30, 50, 160, 255))
    for x in range(4, 8):
        for y in range(8):
            base.putpixel((x, y), (220, 200, 70, 255))
    base.save(expected_path)
    Image.new("RGBA", (8, 8), (140, 90, 90, 255)).save(result_path)

    try:
        compare_generated_texture_against_expected(relative_path, output_root, expected_root, 0.1, 0.1)
    except RuntimeError as error:
        assert "diverged from expected reference" in str(error)
    else:
        raise AssertionError("Expected compare_generated_texture_against_expected to reject the divergent texture")


def test_main_processes_textures_concurrently_and_reports_every_result(tmp_path: Path, monkeypatch) -> None:
    import threading

    import qwen_resourcepack_batch as batch

    items = [(Path(f"assets/mod/textures/block/b{i}.png"), tmp_path / f"b{i}.png") for i in range(6)]
    barrier = threading.Barrier(3, timeout=5)
    seen_threads: set[int] = set()

    def fake_process(source, output, *_args) -> bool:
        seen_threads.add(threading.get_ident())
        barrier.wait()  # only passes if three textures are in flight at the same time
        return True

    monkeypatch.setattr(batch, "process_texture", fake_process)
    monkeypatch.setattr(batch, "resolve_items", lambda *a, **k: items)
    monkeypatch.setattr(batch, "discover_textures", lambda root: {})
    monkeypatch.setattr(batch, "ComfyRenderer", lambda *a, **k: None)
    monkeypatch.setattr(batch, "LazyRenderer", lambda r: None)
    monkeypatch.setattr(batch, "copy_pack_icon", lambda *a: None)
    monkeypatch.setattr(batch, "write_pack_metadata", lambda *a: None)
    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"prompt": {}}')
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--output-root", str(tmp_path / "out"), "--workflow", str(workflow), "--comfy-root", str(tmp_path),
         "--source-root", str(tmp_path), "--workers", "3"],
    )

    assert batch.main() == 0
    assert len(seen_threads) == 3
