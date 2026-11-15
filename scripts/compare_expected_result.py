#!/usr/bin/env python3
"""Compare a generated texture against an expected texture from the reference pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("texture"))
    parser.add_argument("--output-root", type=Path, default=Path("output/resourcepack_1024x_pbr"))
    parser.add_argument("--relative-path", type=Path, required=True)
    parser.add_argument("--expected-path", type=Path)
    parser.add_argument("--result-path", type=Path)
    return parser.parse_args()


def resolve_expected_path(reference_root: Path, relative_path: Path, expected_path: Path | None = None) -> Path:
    return expected_path if expected_path is not None else reference_root / relative_path


def resolve_result_path(output_root: Path, relative_path: Path, result_path: Path | None = None) -> Path:
    return result_path if result_path is not None else output_root / relative_path


def load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA")


def alpha_similarity(expected: Image.Image, result: Image.Image) -> float:
    expected_alpha = np.asarray(expected.getchannel("A"), dtype=np.float32) / 255.0
    result_alpha = np.asarray(result.getchannel("A"), dtype=np.float32) / 255.0
    return float(1.0 - np.mean(np.abs(expected_alpha - result_alpha)))


def luminance_similarity(expected: Image.Image, result: Image.Image) -> float:
    expected_gray = np.asarray(expected.convert("L"), dtype=np.float32) / 255.0
    result_gray = np.asarray(result.convert("L"), dtype=np.float32) / 255.0
    return float(1.0 - np.mean(np.abs(expected_gray - result_gray)))


def color_similarity(expected: Image.Image, result: Image.Image) -> float:
    expected_rgb = np.asarray(expected.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    result_rgb = np.asarray(result.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    expected_mean = expected_rgb.mean(axis=0)
    result_mean = result_rgb.mean(axis=0)
    return float(1.0 - np.mean(np.abs(expected_mean - result_mean)) / 255.0)


def edge_similarity(expected: Image.Image, result: Image.Image) -> float:
    expected_gray = np.asarray(expected.convert("L"), dtype=np.uint8)
    result_gray = np.asarray(result.convert("L"), dtype=np.uint8)
    expected_edges = cv2.Canny(expected_gray, 40, 120) > 0
    result_edges = cv2.Canny(result_gray, 40, 120) > 0
    union = np.logical_or(expected_edges, result_edges)
    if not union.any():
        return 1.0
    intersection = np.logical_and(expected_edges, result_edges)
    return float(intersection.sum() / union.sum())


def compare_expected_to_result(expected_path: Path, result_path: Path) -> dict[str, float | str | list[int]]:
    expected = load_rgba(expected_path)
    result = load_rgba(result_path)
    resized_expected = expected.resize(result.size, Image.Resampling.NEAREST)
    metrics = {
        "expected_path": str(expected_path),
        "result_path": str(result_path),
        "result_size": [result.width, result.height],
        "alpha_similarity": round(alpha_similarity(resized_expected, result), 6),
        "luminance_similarity": round(luminance_similarity(resized_expected, result), 6),
        "color_similarity": round(color_similarity(resized_expected, result), 6),
        "edge_similarity": round(edge_similarity(resized_expected, result), 6),
    }
    metrics["composite_score"] = round(
        (
            metrics["alpha_similarity"] * 0.15
            + metrics["luminance_similarity"] * 0.35
            + metrics["color_similarity"] * 0.2
            + metrics["edge_similarity"] * 0.3
        ),
        6,
    )
    return metrics


def main() -> int:
    args = parse_args()
    expected_path = resolve_expected_path(args.reference_root, args.relative_path, args.expected_path)
    result_path = resolve_result_path(args.output_root, args.relative_path, args.result_path)
    if not expected_path.is_file():
        raise SystemExit(f"Expected texture not found: {expected_path}")
    if not result_path.is_file():
        raise SystemExit(f"Result texture not found: {result_path}")
    print(json.dumps(compare_expected_to_result(expected_path, result_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
