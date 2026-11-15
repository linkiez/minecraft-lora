import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from purebdcraft_creative import parse_category_loras
from qwen_comfy_renderer import ComfyRenderer, build_workflow, latent_size_for, resolve_upscale_model_name


def test_build_workflow_sets_texture_specific_inputs() -> None:
    template = {
        "prompt": {
            "2": {"inputs": {"strength_model": 1.0}},
            "7": {"inputs": {"image": "old.png"}},
            "9": {"inputs": {"prompt": "old"}},
            "14": {"inputs": {"seed": 0, "steps": 4}},
            "16": {"inputs": {"filename_prefix": "old"}},
        }
    }

    workflow = build_workflow(
        template,
        input_name="frame.png",
        prompt="redesign item",
        output_prefix="batch/abc",
        seed=117,
        steps=12,
        lora_strength=0.8,
        denoise=0.65,
    )

    assert workflow["prompt"]["2"]["inputs"]["strength_model"] == 0.8
    assert workflow["prompt"]["7"]["inputs"]["image"] == "frame.png"
    assert workflow["prompt"]["9"]["inputs"]["prompt"] == "redesign item"
    assert workflow["prompt"]["14"]["inputs"] == {"seed": 117, "steps": 12, "denoise": 0.65}
    assert workflow["prompt"]["16"]["inputs"]["filename_prefix"] == "batch/abc"


def test_build_workflow_selects_a_category_specific_lora() -> None:
    template = {
        "prompt": {
            "2": {"inputs": {"lora_name": "general.safetensors", "strength_model": 1.0}},
            "7": {"inputs": {"image": "old.png"}},
            "9": {"inputs": {"prompt": "old"}},
            "14": {"inputs": {"seed": 0, "steps": 4}},
            "16": {"inputs": {"filename_prefix": "old"}},
        }
    }

    workflow = build_workflow(
        template,
        input_name="frame.png",
        prompt="redesign block",
        output_prefix="batch/abc",
        seed=117,
        steps=12,
        lora_strength=0.8,
        lora_name="minecraft-block-qwen-edit-2511-r16.safetensors",
    )

    assert workflow["prompt"]["2"]["inputs"]["lora_name"] == "minecraft-block-qwen-edit-2511-r16.safetensors"


def test_category_specific_loras_are_enabled_by_default() -> None:
    assert parse_category_loras([]) == {
        "block": "minecraft-block-qwen-edit-2511-r16.safetensors",
        "item": "minecraft-item-qwen-edit-2511-r16.safetensors",
    }


def test_build_workflow_uses_purebdcraft_style_reference_as_second_image() -> None:
    template = {
        "prompt": {
            "2": {"inputs": {"strength_model": 1.0}},
            "7": {"inputs": {"image": "old.png"}},
            "8": {"inputs": {"image": ["7", 0]}},
            "9": {"inputs": {"image1": ["8", 0], "prompt": "old"}},
            "10": {"inputs": {"image1": ["8", 0], "prompt": ""}},
            "13": {"inputs": {"pixels": ["8", 0]}},
            "14": {"inputs": {"seed": 0, "steps": 4}},
            "16": {"inputs": {"filename_prefix": "old"}},
        }
    }

    workflow = build_workflow(
        template,
        input_name="source.png",
        prompt="redesign item",
        output_prefix="batch/abc",
        seed=117,
        steps=12,
        lora_strength=0.8,
        style_reference_name="purebdcraft.png",
    )

    assert workflow["prompt"]["19"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "purebdcraft.png"},
    }
    assert workflow["prompt"]["20"] == {
        "class_type": "FluxKontextImageScale",
        "inputs": {"image": ["19", 0]},
    }
    assert workflow["prompt"]["9"]["inputs"]["image1"] == ["8", 0]
    assert workflow["prompt"]["9"]["inputs"]["image2"] == ["20", 0]
    assert workflow["prompt"]["10"]["inputs"]["image2"] == ["20", 0]
    assert workflow["prompt"]["21"] == {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    }
    assert workflow["prompt"]["14"]["inputs"]["latent_image"] == ["21", 0]
    assert workflow["prompt"]["14"]["inputs"]["denoise"] == 1.0


def test_build_workflow_upscales_original_before_creative_generation() -> None:
    template = {
        "prompt": {
            "2": {"inputs": {"strength_model": 1.0}},
            "7": {"inputs": {"image": "old.png"}},
            "8": {"inputs": {"image": ["7", 0]}},
            "9": {"inputs": {"prompt": "old"}},
            "13": {"inputs": {"pixels": ["8", 0]}},
            "14": {"inputs": {"seed": 0, "steps": 4}},
            "16": {"inputs": {"filename_prefix": "old", "images": ["15", 0]}},
        }
    }

    workflow = build_workflow(
        template,
        input_name="frame.png",
        prompt="redesign block",
        output_prefix="batch/abc",
        seed=117,
        steps=12,
        lora_strength=0.8,
        upscale_model_name="4xNomos8kDAT.pth",
    )

    assert workflow["prompt"]["17"]["class_type"] == "UpscaleModelLoader"
    assert workflow["prompt"]["17"]["inputs"]["model_name"] == "4xNomos8kDAT.pth"
    assert workflow["prompt"]["18"]["inputs"] == {"upscale_model": ["17", 0], "image": ["7", 0]}
    assert workflow["prompt"]["8"]["inputs"]["image"] == ["18", 0]
    assert workflow["prompt"]["13"]["inputs"]["pixels"] == ["8", 0]
    assert workflow["prompt"]["16"]["inputs"]["images"] == ["15", 0]


def test_comfy_renderer_uploads_native_texture_for_model_preprocessing(tmp_path: Path) -> None:
    template = {
        "prompt": {
            "2": {"inputs": {"strength_model": 1.0}},
            "7": {"inputs": {"image": "old.png"}},
            "9": {"inputs": {"prompt": "old"}},
            "14": {"inputs": {"seed": 0, "steps": 4}},
            "16": {"inputs": {"filename_prefix": "old"}},
        }
    }
    renderer = ComfyRenderer(
        "http://example.test",
        tmp_path,
        template,
        steps=12,
        lora_strength=0.8,
        category_loras={},
        upscale_model_name=None,
    )
    output_path = tmp_path / "output" / "result.png"
    output_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 32), "green").save(output_path)
    uploaded_sizes: list[tuple[int, int]] = []
    uploaded_colors: list[set[tuple[int, int, int]]] = []

    def request_json(path: str, _: dict | None = None) -> dict:
        if path == "/prompt":
            with Image.open(next(renderer.input_root.glob("*.png"))) as uploaded:
                uploaded_sizes.append(uploaded.size)
                uploaded_colors.append(set(uploaded.convert("RGB").get_flattened_data()))
            return {"prompt_id": "test-prompt"}
        raise AssertionError(f"Unexpected request path: {path}")

    renderer.request_json = request_json
    renderer.wait_for_history = lambda _: {
        "outputs": {"16": {"images": [{"filename": "result.png"}]}}
    }

    source = Image.new("RGB", (8, 4), "red")
    source.paste(Image.new("RGB", (4, 4), "blue"), (4, 0))
    rendered = renderer(source, "prompt", "texture/frame")

    assert uploaded_sizes == [(8, 4)]
    assert uploaded_colors == [{(255, 0, 0), (0, 0, 255)}]
    assert rendered.size == (64, 32)


def test_comfy_renderer_submits_category_style_reference_as_image2(tmp_path: Path) -> None:
    template = {
        "prompt": {
            "2": {"inputs": {"strength_model": 1.0}},
            "7": {"inputs": {"image": "old.png"}},
            "8": {"inputs": {"image": ["7", 0]}},
            "9": {"inputs": {"image1": ["8", 0], "prompt": "old"}},
            "10": {"inputs": {"image1": ["8", 0], "prompt": ""}},
            "13": {"inputs": {"pixels": ["8", 0]}},
            "14": {"inputs": {"seed": 0, "steps": 4}},
            "16": {"inputs": {"filename_prefix": "old"}},
        }
    }
    style_root = tmp_path / "styles"
    item_root = style_root / "qwen_edit_pairs_item" / "musubi_target"
    item_root.mkdir(parents=True)
    Image.new("RGB", (16, 16), "blue").save(item_root / "purebdcraft.png")
    renderer = ComfyRenderer(
        "http://example.test",
        tmp_path,
        template,
        steps=12,
        lora_strength=0.8,
        category_loras={},
        style_reference_root=style_root,
    )
    output_path = tmp_path / "output" / "result.png"
    output_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "green").save(output_path)
    submitted_workflows: list[dict] = []

    def request_json(path: str, payload: dict | None = None) -> dict:
        if path == "/prompt":
            assert payload is not None
            style_name = payload["prompt"]["19"]["inputs"]["image"]
            assert (tmp_path / "input" / style_name).is_file()
            submitted_workflows.append(payload)
            return {"prompt_id": "test-prompt"}
        raise AssertionError(f"Unexpected request path: {path}")

    renderer.request_json = request_json
    renderer.wait_for_history = lambda _: {
        "outputs": {"16": {"images": [{"filename": "result.png"}]}}
    }

    renderer(Image.new("RGB", (8, 8), "red"), "redesign", "assets/mod/textures/item/test.png")

    positive_inputs = submitted_workflows[0]["prompt"]["9"]["inputs"]
    assert positive_inputs["image2"] == ["20", 0]
    assert "Image 2 is the PureBDCraft style reference" in positive_inputs["prompt"]
    assert "unique palette, markings, symbols, and details from Image 1" in positive_inputs["prompt"]
    assert submitted_workflows[0]["prompt"]["14"]["inputs"]["latent_image"] == ["21", 0]


def test_resolve_upscale_model_does_not_enable_installed_model_implicitly(tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "upscale_models"
    models_root.mkdir(parents=True)
    (models_root / "4xNomos8kDAT.pth").touch()

    assert resolve_upscale_model_name(tmp_path, "") is None


def test_latent_size_keeps_aspect_ratio_at_one_megapixel() -> None:
    assert latent_size_for((16, 16)) == (1024, 1024)
    assert latent_size_for((16, 32)) == (720, 1456)
    assert latent_size_for((0, 16)) == (1024, 1024)


def test_build_workflow_uses_requested_latent_size_with_style_reference() -> None:
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

    workflow = build_workflow(
        template, "in.png", "p", "out", 1, 12, 0.8, 0.25,
        style_reference_name="style.png", latent_size=(720, 1456),
    )

    assert workflow["prompt"]["21"]["inputs"] == {"width": 720, "height": 1456, "batch_size": 1}
    assert workflow["prompt"]["14"]["inputs"]["denoise"] == 1.0


def test_comfy_renderer_removes_comfyui_output_and_temp_inputs(tmp_path: Path) -> None:
    template = {
        "prompt": {
            "2": {"inputs": {"strength_model": 1.0}},
            "7": {"inputs": {"image": "old.png"}},
            "8": {"inputs": {"image": ["7", 0]}},
            "9": {"inputs": {"image1": ["8", 0], "prompt": "old"}},
            "10": {"inputs": {"image1": ["8", 0], "prompt": ""}},
            "13": {"inputs": {"pixels": ["8", 0]}},
            "14": {"inputs": {"seed": 0, "steps": 4}},
            "16": {"inputs": {"filename_prefix": "old"}},
        }
    }
    renderer = ComfyRenderer(
        "http://example.test", tmp_path, template,
        steps=12, lora_strength=0.8, category_loras={}, style_reference_root=None,
    )
    output_path = tmp_path / "output" / "result.png"
    output_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "green").save(output_path)
    renderer.request_json = lambda path, payload=None: {"prompt_id": "p"}
    renderer.wait_for_history = lambda _: {"outputs": {"16": {"images": [{"filename": "result.png"}]}}}

    result = renderer(Image.new("RGB", (8, 8), "red"), "redesign", "assets/mod/textures/item/test.png")

    assert result.size == (64, 64)
    assert not output_path.exists()
    assert list((tmp_path / "input" / "resourcepack_batch").iterdir()) == []


def test_comfy_renderer_cleans_up_when_plan_is_not_available(tmp_path: Path) -> None:
    renderer = ComfyRenderer(
        "http://example.test", tmp_path, {"prompt": {}},
        steps=12, lora_strength=0.8, category_loras={}, style_reference_root=None,
    )

    with pytest.raises(ValueError):
        renderer(Image.new("RGB", (8, 8), "red"), "x", "assets/mod/textures/gui/test.png")

    assert list((tmp_path / "input" / "resourcepack_batch").iterdir()) == []


def test_build_workflow_sets_negative_prompt() -> None:
    template = {
        "prompt": {
            "2": {"inputs": {}},
            "7": {"inputs": {}},
            "9": {"inputs": {}},
            "10": {"inputs": {"prompt": ""}},
            "14": {"inputs": {}},
            "16": {"inputs": {}},
        }
    }

    workflow = build_workflow(template, "in.png", "p", "out", 1, 12, 0.8, negative_prompt="pixel grid")

    assert workflow["prompt"]["10"]["inputs"]["prompt"] == "pixel grid"


def test_comfy_renderer_uses_unique_output_prefix_per_submission(tmp_path: Path) -> None:
    template = {
        "prompt": {
            "2": {"inputs": {}},
            "7": {"inputs": {}},
            "8": {"inputs": {}},
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
    prefixes: list[str] = []

    def request_json(path: str, payload: dict | None = None) -> dict:
        prefixes.append(payload["prompt"]["16"]["inputs"]["filename_prefix"])
        Image.new("RGB", (8, 8), "green").save(output_path)
        return {"prompt_id": "p"}

    renderer.request_json = request_json
    renderer.wait_for_history = lambda _: {"outputs": {"16": {"images": [{"filename": "result.png"}]}}}

    for _ in range(2):
        renderer(Image.new("RGB", (8, 8), "red"), "x", "assets/mod/textures/item/test.png")

    assert prefixes[0] != prefixes[1]



def test_build_workflow_chains_speed_lora_and_disables_cfg() -> None:
    template = {
        "prompt": {
            "2": {"inputs": {}},
            "3": {"inputs": {"model": ["2", 0]}},
            "7": {"inputs": {}},
            "9": {"inputs": {}},
            "10": {"inputs": {}},
            "14": {"inputs": {"cfg": 4.0, "positive": ["11", 0], "negative": ["12", 0]}},
            "16": {"inputs": {}},
        }
    }

    fast = build_workflow(template, "in.png", "p", "out", 1, 4, 0.8, speed_lora_name="lightning.safetensors", cfg=1.0)
    normal = build_workflow(template, "in.png", "p", "out", 1, 12, 0.8)

    assert fast["prompt"]["22"]["inputs"]["model"] == ["2", 0]
    assert fast["prompt"]["22"]["inputs"]["lora_name"] == "lightning.safetensors"
    assert fast["prompt"]["3"]["inputs"]["model"] == ["22", 0]
    assert fast["prompt"]["14"]["inputs"]["cfg"] == 1.0
    assert "22" not in normal["prompt"]
    assert normal["prompt"]["3"]["inputs"]["model"] == ["2", 0]
    assert normal["prompt"]["14"]["inputs"]["cfg"] == 4.0


def test_comfy_renderer_with_speed_lora_samples_from_full_noise_without_cfg(tmp_path: Path) -> None:
    template = {
        "prompt": {
            "2": {"inputs": {}},
            "3": {"inputs": {"model": ["2", 0]}},
            "7": {"inputs": {}},
            "8": {"inputs": {}},
            "9": {"inputs": {}},
            "10": {"inputs": {}},
            "14": {"inputs": {"cfg": 4.0, "positive": ["11", 0], "negative": ["12", 0]}},
            "16": {"inputs": {}},
        }
    }
    renderer = ComfyRenderer(
        "http://example.test", tmp_path, template,
        steps=4, lora_strength=0.8, category_loras={}, style_reference_root=None,
        speed_lora_name="lightning.safetensors",
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

    renderer(Image.new("RGB", (8, 8), "red"), "x", "assets/mod/textures/block/test.png")

    nodes = submitted[0]["prompt"]
    assert nodes["14"]["inputs"]["denoise"] == 1.0
    assert nodes["14"]["inputs"]["cfg"] == 1.0
    assert nodes["14"]["inputs"]["steps"] == 4
    assert nodes["22"]["inputs"]["lora_name"] == "lightning.safetensors"


def test_build_workflow_without_cfg_reuses_positive_conditioning_as_negative() -> None:
    template = {
        "prompt": {
            "2": {"inputs": {}},
            "3": {"inputs": {"model": ["2", 0]}},
            "7": {"inputs": {}},
            "9": {"inputs": {}},
            "10": {"inputs": {}},
            "14": {"inputs": {"cfg": 4.0, "positive": ["11", 0], "negative": ["12", 0]}},
            "16": {"inputs": {}},
        }
    }

    unguided = build_workflow(template, "in.png", "p", "out", 1, 4, 0.8, speed_lora_name="l.safetensors", cfg=1.0)
    guided = build_workflow(template, "in.png", "p", "out", 1, 12, 0.8, cfg=4.0)

    assert unguided["prompt"]["14"]["inputs"]["negative"] == ["11", 0]
    assert guided["prompt"]["14"]["inputs"]["negative"] == ["12", 0]
