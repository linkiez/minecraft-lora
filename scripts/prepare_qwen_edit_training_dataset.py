#!/usr/bin/env python3
"""Prepare paired source/target images for Qwen-Image-Edit LoRA training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from texture_frames import load_texture_metadata, split_animation_frames

TEXTURE_CATEGORIES = ("item", "block", "entity", "gui", "environment", "particle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, default=Path("datasets/lora_prep/pairs.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/qwen_edit_pairs"))
    parser.add_argument("--category", action="append", choices=TEXTURE_CATEGORIES)
    return parser.parse_args()


def category_for(relative_path: Path) -> str:
    return next((candidate for candidate in TEXTURE_CATEGORIES if candidate in relative_path.parts), "other")


def instruction_for(relative_path: Path, frame_index: int | None = None) -> str:
    texture_type = category_for(relative_path)
    name = relative_path.stem.replace("_", " ").replace("-", " ")
    frame = f" Animation frame {frame_index}." if frame_index is not None else ""
    if texture_type == "entity":
        return (
            f"Creatively restyle this Minecraft entity UV atlas named {name} in a polished cartoon game style. "
            "Treat every face, clothing, limb, and hair island as an independent atlas region; preserve its position, "
            "orientation, silhouette, and transparent background. Improve material detail and lighting only within each "
            f"existing island. Do not assemble a character, move regions, add scenery, text, pixel grids, or duplicates.{frame}"
        )
    if texture_type == "gui":
        return (
            f"Creatively restyle this Minecraft GUI atlas named {name} in a polished cartoon game style. Preserve exact "
            "element placement, dimensions, transparent background, state meaning, and atlas layout. Improve only local "
            f"line work, palette, material detail, and highlights. Do not add text, controls, scenery, borders, or duplicates.{frame}"
        )
    if texture_type == "block":
        return (
            f"Creatively restyle this Minecraft block material named {name} as a seamless polished cartoon game texture. "
            "Preserve its material identity and make all edges tile continuously with matched colors and details. Do not add "
            f"objects, scenery, text, pixel grids, borders, or directional shadows.{frame}"
        )
    if texture_type == "particle":
        return (
            f"Creatively restyle this Minecraft particle sprite named {name} with clear readable cartoon shapes. Preserve "
            "the exact frame layout, silhouette, transparent background, and animation meaning. Do not add scenery, text, "
            f"borders, pixel grids, or duplicates.{frame}"
        )
    if texture_type == "environment":
        return (
            f"Creatively restyle this Minecraft environment texture named {name} in a polished cartoon game style. Preserve "
            "the exact composition, orientation, atlas layout, and any transparent regions. Do not add objects, text, pixel "
            f"grids, borders, or duplicates.{frame}"
        )
    return (
        f"Redesign this Minecraft {texture_type} named {name} in a polished artistic cartoon style. "
        "Use smooth continuous curves, clean antialiased edges, dimensional cel shading, vibrant colors, "
        "and a clear silhouette. Preserve the identity, geometry, orientation, and transparent background. "
        f"Do not use pixel art, square pixels, a voxel grid, dithering, text, or a new background.{frame}"
    )


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        destination.symlink_to(source.resolve())


def load_frames(path: Path) -> list[Image.Image]:
    with Image.open(path) as image:
        source = image.copy()
    return split_animation_frames(source, load_texture_metadata(path))


def frame_path(relative_path: Path, frame_index: int) -> Path:
    return relative_path.with_name(f"{relative_path.stem}__frame_{frame_index:04d}{relative_path.suffix}")


def save_pair(
    control: Image.Image,
    target: Image.Image,
    relative_path: Path,
    output_root: Path,
    frame_index: int,
) -> None:
    output_relative = frame_path(relative_path, frame_index)
    control_path = output_root / "control" / output_relative
    target_path = output_root / "target" / output_relative
    control_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    control.save(control_path)
    target.save(target_path)
    target_path.with_suffix(".txt").write_text(
        instruction_for(relative_path, frame_index) + "\n",
        encoding="utf-8",
    )


def prepare_row(row: dict[str, str], output_root: Path) -> int:
    relative_path = Path(row["rel_key"])
    control_path = Path(row["base_path"])
    target_path = Path(row["resource_path"])
    if not control_path.is_file() or not target_path.is_file():
        return 0

    control_frames = load_frames(control_path)
    target_frames = load_frames(target_path)
    frame_count = max(len(control_frames), len(target_frames))
    if frame_count == 1:
        output_control = output_root / "control" / relative_path
        output_target = output_root / "target" / relative_path
        hardlink(control_path, output_control)
        hardlink(target_path, output_target)
        output_target.with_suffix(".txt").write_text(instruction_for(relative_path) + "\n", encoding="utf-8")
        return 1

    for frame_index in range(frame_count):
        control_index = min(len(control_frames) - 1, frame_index * len(control_frames) // frame_count)
        target_index = min(len(target_frames) - 1, frame_index * len(target_frames) // frame_count)
        save_pair(
            control_frames[control_index],
            target_frames[target_index],
            relative_path,
            output_root,
            frame_index,
        )
    return frame_count


def write_metadata(output_root: Path) -> int:
    records = 0
    metadata_path = output_root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for target_path in sorted((output_root / "target").rglob("*.png")):
            relative_path = target_path.relative_to(output_root / "target")
            control_path = output_root / "control" / relative_path
            caption_path = target_path.with_suffix(".txt")
            if not control_path.is_file() or not caption_path.is_file():
                continue
            digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:16]
            musubi_name = f"{digest}__{relative_path.name}"
            musubi_target = output_root / "musubi_target" / musubi_name
            musubi_control = output_root / "musubi_control" / musubi_name
            hardlink(target_path.resolve(), musubi_target)
            hardlink(control_path.resolve(), musubi_control)
            record = {
                "image_path": str(musubi_target.resolve()),
                "control_path": str(musubi_control.resolve()),
                "caption": caption_path.read_text(encoding="utf-8").strip(),
            }
            metadata.write(json.dumps(record, ensure_ascii=True) + "\n")
            records += 1
    return records


def prepare_dataset(
    pairs_csv: Path,
    output_root: Path,
    allowed_categories: set[str] | None = None,
) -> tuple[int, int]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to mix an existing dataset: {output_root}")
    output_root.mkdir(parents=True)
    generated = skipped = 0
    with pairs_csv.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if allowed_categories is not None and category_for(Path(row["rel_key"])) not in allowed_categories:
                skipped += 1
                continue
            try:
                row_count = prepare_row(row, output_root)
            except (OSError, ValueError):
                row_count = 0
            if row_count:
                generated += row_count
            else:
                skipped += 1
    metadata_records = write_metadata(output_root)
    if metadata_records != generated:
        raise RuntimeError(f"Generated {generated} pairs but wrote {metadata_records} metadata records")
    return generated, skipped


def main() -> None:
    args = parse_args()
    categories = set(args.category) if args.category else None
    generated, skipped = prepare_dataset(args.pairs_csv, args.output_root, categories)
    print(f"Qwen edit dataset complete: pairs={generated} skipped={skipped} output={args.output_root}")


if __name__ == "__main__":
    main()
