#!/usr/bin/env python3
"""Remove a light, low-saturation background connected to image borders."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-lightness", type=int, default=145)
    parser.add_argument("--maximum-chroma", type=int, default=55)
    parser.add_argument("--foreground-chroma", type=int)
    parser.add_argument("--feather-radius", type=float, default=1.2)
    return parser.parse_args()


def connected_background_mask(
    image: np.ndarray,
    minimum_lightness: int,
    maximum_chroma: int,
) -> np.ndarray:
    rgb = image[:, :, :3].astype(np.int16)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    lightness = (maximum + minimum) // 2
    eligible = ((maximum - minimum) <= maximum_chroma) & (lightness >= minimum_lightness)

    component_count, labels = cv2.connectedComponents(eligible.astype(np.uint8), connectivity=8)
    if component_count <= 1:
        return np.zeros(eligible.shape, dtype=bool)

    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    return np.isin(labels, border_labels)


def largest_chromatic_foreground(image: np.ndarray, minimum_chroma: int) -> np.ndarray:
    rgb = image[:, :, :3].astype(np.int16)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (chroma >= minimum_chroma).astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return np.zeros(chroma.shape, dtype=bool)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest_label).astype(np.uint8) * 255
    flood = component.copy()
    flood_mask = np.zeros((component.shape[0] + 2, component.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    filled = component | cv2.bitwise_not(flood)
    return filled.astype(bool)


def remove_background(
    source: Path,
    destination: Path,
    minimum_lightness: int,
    maximum_chroma: int,
    feather_radius: float,
    foreground_chroma: int | None = None,
) -> None:
    with Image.open(source) as opened:
        image = np.asarray(opened.convert("RGBA")).copy()

    if foreground_chroma is None:
        background = connected_background_mask(image, minimum_lightness, maximum_chroma)
        alpha = np.where(background, 0, 255).astype(np.uint8)
    else:
        foreground = largest_chromatic_foreground(image, foreground_chroma)
        alpha = np.where(foreground, 255, 0).astype(np.uint8)
    if feather_radius > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_radius)
    image[:, :, 3] = alpha

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGBA").save(destination)


def main() -> None:
    args = parse_args()
    remove_background(
        args.input,
        args.output,
        args.minimum_lightness,
        args.maximum_chroma,
        args.feather_radius,
        args.foreground_chroma,
    )


if __name__ == "__main__":
    main()
