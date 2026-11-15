#!/usr/bin/env python3
"""Pair every vanilla/mod texture with its counterpart in a reference resource pack.

Writes CSV manifests used to build LoRA training datasets: ``pairs.csv`` (textures present on both sides) plus
``base_only.csv``, ``resourcepack_only.csv``, ``all_base.csv`` and train/validation/test splits of the pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable


@dataclass(frozen=True)
class Entry:
    source: str
    abs_path: Path
    rel_key: str


def iter_png_assets(root: Path, source_name: str) -> Iterable[Entry]:
    if not root.exists():
        return
    for p in root.rglob("*.png"):
        if not p.is_file():
            continue
        parts = p.parts
        if "assets" not in parts:
            continue
        i = parts.index("assets")
        rel = Path(*parts[i:]).as_posix()
        if "/textures/" not in rel:
            continue
        yield Entry(source=source_name, abs_path=p.resolve(), rel_key=rel)


def build_maps(original_assets: Path, mods_assets: Path, resource_assets: Path) -> tuple[dict[str, Entry], dict[str, Entry]]:
    base_map: dict[str, Entry] = {}

    # Base precedence: original first, then mods override
    for e in iter_png_assets(original_assets, "original"):
        base_map[e.rel_key] = e
    for e in iter_png_assets(mods_assets, "mods"):
        base_map[e.rel_key] = e

    resource_map: dict[str, Entry] = {}
    for e in iter_png_assets(resource_assets, "resource_pack"):
        resource_map[e.rel_key] = e

    return base_map, resource_map


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows: list[dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def build_generation_rows(base_map: dict[str, Entry], resource_map: dict[str, Entry]) -> list[dict[str, str]]:
    """One row per base texture, with the reference-pack path when the pack also has it."""
    return [
        {
            "rel_key": rel_key,
            "base_source": base_map[rel_key].source,
            "base_path": str(base_map[rel_key].abs_path),
            "resource_path": str(resource_map[rel_key].abs_path) if rel_key in resource_map else "",
        }
        for rel_key in sorted(base_map)
    ]


PAIR_FIELDS = ["rel_key", "base_source", "base_path", "resource_path"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original-assets", type=Path, default=Path("original/assets"), help="Vanilla assets folder.")
    parser.add_argument("--mods-assets", type=Path, default=Path("mods/assets"), help="Mod assets folder.")
    parser.add_argument(
        "--resource-assets", type=Path, default=Path("texture/assets"), help="Reference resource-pack assets folder."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/lora_prep"), help="Where the CSVs are written.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir
    base_map, resource_map = build_maps(args.original_assets, args.mods_assets, args.resource_assets)

    keys_base = set(base_map)
    keys_rp = set(resource_map)
    common = sorted(keys_base & keys_rp)
    rp_only = sorted(keys_rp - keys_base)
    base_only = sorted(keys_base - keys_rp)

    pair_rows = [
        {
            "rel_key": key,
            "base_source": base_map[key].source,
            "base_path": str(base_map[key].abs_path),
            "resource_path": str(resource_map[key].abs_path),
        }
        for key in common
    ]
    rp_only_rows = [{"rel_key": key, "resource_path": str(resource_map[key].abs_path)} for key in rp_only]
    base_only_rows = [
        {"rel_key": key, "base_source": base_map[key].source, "base_path": str(base_map[key].abs_path)}
        for key in base_only
    ]
    all_base_rows = build_generation_rows(base_map, resource_map)

    write_csv(out_dir / "pairs.csv", pair_rows, PAIR_FIELDS)
    write_csv(out_dir / "resourcepack_only.csv", rp_only_rows, ["rel_key", "resource_path"])
    write_csv(out_dir / "base_only.csv", base_only_rows, ["rel_key", "base_source", "base_path"])
    write_csv(out_dir / "all_base.csv", all_base_rows, PAIR_FIELDS)

    train_rows, val_rows, test_rows = split_rows(pair_rows)
    write_csv(out_dir / "pairs_train.csv", train_rows, PAIR_FIELDS)
    write_csv(out_dir / "pairs_val.csv", val_rows, PAIR_FIELDS)
    write_csv(out_dir / "pairs_test.csv", test_rows, PAIR_FIELDS)

    summary = {
        "base_total": len(keys_base),
        "resource_total": len(keys_rp),
        "pairs_total": len(common),
        "resourcepack_only_total": len(rp_only),
        "base_only_total": len(base_only),
        "all_base_total": len(all_base_rows),
        "output_dir": str(out_dir.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
