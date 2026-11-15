import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen_comfy_renderer import ComfyRenderer


def submitted_prompt_text(tmp_path: Path, prompt: str) -> str:
    template = {
        "prompt": {
            "2": {"inputs": {}},
            "7": {"inputs": {}},
            "9": {"inputs": {}},
            "10": {"inputs": {}},
            "14": {"inputs": {}},
            "16": {"inputs": {}},
        }
    }
    renderer = ComfyRenderer(
        "http://example.test", tmp_path, template,
        steps=12, lora_strength=0.8, category_loras={}, style_reference_root=None,
    )
    output_path = tmp_path / "output" / "result.png"
    output_path.parent.mkdir(parents=True)
    submitted: list[dict] = []

    def request_json(path: str, payload: dict | None = None) -> dict:
        submitted.append(payload)
        Image.new("RGB", (8, 8), "green").save(output_path)
        return {"prompt_id": "p"}

    renderer.request_json = request_json
    renderer.wait_for_history = lambda _: {"outputs": {"16": {"images": [{"filename": "result.png"}]}}}

    renderer(Image.new("RGB", (8, 8), "red"), prompt, "assets/mod/textures/block/test.png")

    return submitted[0]["prompt"]["9"]["inputs"]["prompt"]


def test_comfy_renderer_submits_the_prompt_given_by_the_caller(tmp_path: Path) -> None:
    assert submitted_prompt_text(tmp_path, "custom instruction with background clause") == (
        "custom instruction with background clause"
    )


def test_comfy_renderer_falls_back_to_the_plan_prompt_when_the_caller_gives_none(tmp_path: Path) -> None:
    assert "Minecraft block" in submitted_prompt_text(tmp_path, "")
