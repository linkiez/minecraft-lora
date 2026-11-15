import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from purebdcraft_creative import (
    PureBDCraftCreativeStrategy,
    denoise_for,
    prompt_for,
    select_style_reference,
)


def test_purebdcraft_strategy_builds_complete_item_generation_plan(tmp_path: Path) -> None:
    strategy = PureBDCraftCreativeStrategy(
        reference_root=tmp_path,
        category_loras={"item": "item-style.safetensors"},
    )

    plan = strategy.plan_for(
        Path("assets/mod/textures/item/enchanted_book.png"),
        Image.new("RGBA", (16, 16), "purple"),
    )

    assert plan.category == "item"
    assert plan.denoise == 0.8
    assert plan.lora_name == "item-style.safetensors"
    assert "PureBDCraft" in plan.prompt
    assert "enchanted book" in plan.prompt


def test_purebdcraft_strategy_rejects_non_creative_texture_categories(tmp_path: Path) -> None:
    strategy = PureBDCraftCreativeStrategy(reference_root=tmp_path, category_loras={})

    assert strategy.plan_for(Path("assets/mod/textures/entity/cow.png"), Image.new("RGB", (16, 16))) is None


def test_select_style_reference_is_deterministic_and_category_specific(tmp_path: Path) -> None:
    item_root = tmp_path / "qwen_edit_pairs_item" / "musubi_target"
    block_root = tmp_path / "qwen_edit_pairs_block" / "musubi_target"
    item_root.mkdir(parents=True)
    block_root.mkdir(parents=True)
    item_references = [item_root / "amulet.png", item_root / "sword.png"]
    block_reference = block_root / "stone.png"
    for path in [*item_references, block_reference]:
        Image.new("RGB", (8, 8), "red").save(path)

    first = select_style_reference(tmp_path, "item", "assets/mod/textures/item/test.png")
    second = select_style_reference(tmp_path, "item", "assets/mod/textures/item/test.png")

    assert first == second
    assert first in item_references
    assert select_style_reference(tmp_path, "block", "assets/mod/textures/block/test.png") == block_reference


def test_select_style_reference_uses_nearest_paired_control(tmp_path: Path) -> None:
    dataset_root = tmp_path / "qwen_edit_pairs_item"
    control_root = dataset_root / "musubi_control"
    target_root = dataset_root / "musubi_target"
    control_root.mkdir(parents=True)
    target_root.mkdir(parents=True)
    book_name = "0001__enchanted_book.png"
    door_name = "0002__wooden_door.png"
    book = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    book.paste(Image.new("RGBA", (10, 8), (70, 30, 120, 255)), (3, 5))
    door = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    door.paste(Image.new("RGBA", (5, 15), (110, 70, 30, 255)), (6, 1))
    book.save(control_root / book_name)
    door.save(control_root / door_name)
    Image.new("RGB", (64, 64), "purple").save(target_root / book_name)
    Image.new("RGB", (64, 64), "brown").save(target_root / door_name)

    selected = select_style_reference(
        tmp_path,
        "item",
        "assets/mod/textures/item/enchanted_book.png",
        source_image=book,
    )

    assert selected == target_root / book_name


def test_block_prompt_demands_full_frame_tile_structure() -> None:
    prompt = prompt_for(Path("assets/mod/textures/block/oak_planks.png"))

    assert prompt.startswith("Creatively restyle this Minecraft block material named oak planks")
    assert "tile continuously" in prompt
    assert "flat block face texture" in prompt
    assert "edge to edge" in prompt
    assert "same occupancy map" in prompt
    assert "never draw an isolated object" in prompt


def test_item_prompt_demands_inventory_readability() -> None:
    prompt = prompt_for(Path("assets/mod/textures/item/diamond_sword.png"))

    assert "inventory icon" in prompt
    assert "Keep the item centered" in prompt
    assert "Do not add scenery" in prompt


def test_particle_prompt_preserves_compact_vfx_sprite_behavior() -> None:
    prompt = prompt_for(Path("assets/mod/textures/particle/spark.png"))

    assert "VFX sprite" in prompt
    assert "Keep the sprite simple" in prompt


def test_environment_prompt_demands_seamless_surface_behavior() -> None:
    prompt = prompt_for(Path("assets/mod/textures/environment/fog.png"))

    assert "seamless stylized game surface" in prompt
    assert "Do not add text" in prompt


def test_denoise_for_follows_category_profile() -> None:
    assert denoise_for(Path("assets/mod/textures/block/oak_planks.png")) == 0.9
    assert denoise_for(Path("assets/mod/textures/item/diamond.png")) == 0.8
    assert denoise_for(Path("assets/mod/textures/particle/spark.png")) == 0.5
    assert denoise_for(Path("assets/mod/textures/environment/fog.png")) == 0.45


def test_block_plan_carries_negative_prompt_against_objects_and_pixel_grid(tmp_path: Path) -> None:
    strategy = PureBDCraftCreativeStrategy(reference_root=None, category_loras={})

    plan = strategy.plan_for(Path("assets/mod/textures/block/stone.png"), Image.new("RGB", (16, 16)))

    assert plan is not None
    assert "pixel grid" in plan.negative_prompt
    assert "isolated object" in plan.negative_prompt


def test_block_plan_does_not_use_a_style_reference_image(tmp_path: Path) -> None:
    block_root = tmp_path / "qwen_edit_pairs_block"
    for folder in ("musubi_control", "musubi_target"):
        (block_root / folder).mkdir(parents=True)
        Image.new("RGB", (16, 16), "gray").save(block_root / folder / "abc__stone.png")
    strategy = PureBDCraftCreativeStrategy(reference_root=tmp_path, category_loras={})

    plan = strategy.plan_for(Path("assets/mod/textures/block/stone.png"), Image.new("RGB", (16, 16), "gray"))

    assert plan is not None
    assert plan.style_reference is None
    assert plan.denoise == 0.9
