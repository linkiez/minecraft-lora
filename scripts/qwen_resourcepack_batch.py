#!/usr/bin/env python3
"""Generate a complete Qwen GGUF LoRA resource pack through the ComfyUI API."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from compare_expected_result import compare_expected_to_result
from purebdcraft_creative import CREATIVE_TEXTURE_CATEGORIES, parse_category_loras
from qwen_comfy_renderer import ComfyRenderer, LazyRenderer, resolve_upscale_model_name
from qwen_texture_processing import process_texture, resolve_items
from resourcepack_layout import copy_pack_icon, discover_textures, write_pack_metadata

DEFAULT_GENERATION_RESOLUTION = 2048
RUN_LOCK_FILE = ".qwen_resourcepack_batch.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("texture"))
    parser.add_argument("--expected-reference-root", type=Path, default=Path("texture"))
    parser.add_argument(
        "--style-reference-root",
        type=Path,
        default=Path("datasets"),
        help="Root containing paired PureBDCraft target datasets used as Qwen image2 style references.",
    )
    parser.add_argument("--workflow", type=Path, default=Path("configs/qwen_gguf_lora_preflight.json"))
    parser.add_argument("--comfy-root", type=Path, default=Path("tools/ComfyUI"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8189")
    parser.add_argument("--resolution", type=int, default=DEFAULT_GENERATION_RESOLUTION)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--speed-lora",
        default="",
        help="Step-distillation LoRA filename in ComfyUI models/loras (e.g. the Qwen-Image-Edit Lightning 4-step "
        "LoRA); use together with --steps 4. Runs without CFG.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Textures processed concurrently, so CPU post-processing overlaps with the next GPU render.",
    )
    parser.add_argument(
        "--upscale-model",
        default="",
        help="Optional ComfyUI upscale model filename from tools/ComfyUI/models/upscale_models.",
    )
    parser.add_argument("--lora-strength", type=float, default=0.8)
    parser.add_argument(
        "--category-lora",
        action="append",
        default=[],
        metavar="CATEGORY=FILENAME",
        help="Override the workflow LoRA filename for a creative texture category.",
    )
    parser.add_argument("--normal-strength", type=float, default=2.0)
    parser.add_argument("--max-textures", type=int, default=0)
    parser.add_argument("--texture", action="append", default=[], help="Relative texture path to process")
    parser.add_argument(
        "--only-qwen",
        action="store_true",
        help="Process only creative Qwen categories (item, block, particle, environment).",
    )
    parser.add_argument(
        "--only-category",
        choices=CREATIVE_TEXTURE_CATEGORIES,
        help="Process only one creative category (item, block, particle, environment).",
    )
    parser.add_argument(
        "--compare-expected",
        action="store_true",
        help="Compare generated textures against matching expected textures under --expected-reference-root.",
    )
    parser.add_argument(
        "--min-expected-edge-similarity",
        type=float,
        default=0.0,
        help="Optional minimum edge similarity required when --compare-expected is enabled.",
    )
    parser.add_argument(
        "--min-expected-composite-score",
        type=float,
        default=0.0,
        help="Optional minimum composite score required when --compare-expected is enabled.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compare_generated_texture_against_expected(
    relative_path: Path,
    output_root: Path,
    expected_reference_root: Path,
    min_edge_similarity: float,
    min_composite_score: float,
) -> dict[str, float | str | list[int]] | None:
    expected_path = expected_reference_root / relative_path
    result_path = output_root / relative_path
    if not expected_path.is_file() or not result_path.is_file():
        return None
    metrics = compare_expected_to_result(expected_path, result_path)
    metrics_path = result_path.with_suffix(".expected_compare.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    edge_similarity = float(metrics["edge_similarity"])
    composite_score = float(metrics["composite_score"])
    print(
        "Expected comparison: "
        f"path={relative_path} edge={edge_similarity:.6f} composite={composite_score:.6f} "
        f"luminance={float(metrics['luminance_similarity']):.6f} color={float(metrics['color_similarity']):.6f}",
        flush=True,
    )
    if edge_similarity < min_edge_similarity or composite_score < min_composite_score:
        raise RuntimeError(
            "Generated texture diverged from expected reference: "
            f"{relative_path} edge={edge_similarity:.6f} composite={composite_score:.6f}"
        )
    return metrics


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def output_generation_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / RUN_LOCK_FILE
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            existing_pid = None
        if existing_pid is not None and process_is_running(existing_pid):
            raise RuntimeError(
                f"Another qwen_resourcepack_batch run is already writing to {output_root} (pid {existing_pid})."
            )
        lock_path.unlink(missing_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another qwen_resourcepack_batch run is already writing to {output_root}.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    reference_root = args.reference_root.resolve()
    expected_reference_root = args.expected_reference_root.resolve()
    style_reference_root = args.style_reference_root.resolve()
    comfy_root = args.comfy_root.resolve()
    template = json.loads(args.workflow.read_text(encoding="utf-8"))
    category_loras = parse_category_loras(args.category_lora)
    upscale_model_name = resolve_upscale_model_name(comfy_root, args.upscale_model)
    print(
        "Creative input preprocessing: Qwen FluxKontextImageScale"
        f"{f' + {upscale_model_name}' if upscale_model_name is not None else ''}",
        flush=True,
    )
    renderer = LazyRenderer(
        ComfyRenderer(
            args.api_url,
            comfy_root,
            template,
            args.steps,
            args.lora_strength,
            category_loras,
            upscale_model_name,
            style_reference_root,
            args.speed_lora or None,
        )
    )
    discovered = discover_textures(source_root)
    items = resolve_items(discovered, args.texture, args.only_qwen, args.only_category, reference_root)
    if args.max_textures > 0:
        items = items[: args.max_textures]
    with output_generation_lock(output_root):
        write_pack_metadata(output_root, args.resolution)
        copy_pack_icon(source_root, output_root)

        generated = skipped = failed = 0

        def run_item(relative_path: Path, source_path: Path) -> bool:
            changed = process_texture(
                source_path,
                output_root / relative_path,
                output_root,
                reference_root,
                args.resolution,
                args.normal_strength,
                renderer,
                args.overwrite,
            )
            if args.compare_expected and changed:
                compare_generated_texture_against_expected(
                    relative_path,
                    output_root,
                    expected_reference_root,
                    args.min_expected_edge_similarity,
                    args.min_expected_composite_score,
                )
            return changed

        # ComfyUI serialises renders on the GPU, so with several workers one texture is always queued while
        # another is being post-processed on the CPU; the GPU no longer idles between textures.
        executor = ThreadPoolExecutor(max_workers=max(1, args.workers))
        try:
            futures = {executor.submit(run_item, path, source): path for path, source in items}
            for index, future in enumerate(as_completed(futures), start=1):
                relative_path = futures[future]
                try:
                    changed = future.result()
                    generated += int(changed)
                    skipped += int(not changed)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    failed += 1
                    print(f"Texture failed: {relative_path}: {error}", flush=True)
                print(
                    f"[{index}/{len(items)}] generated={generated} skipped={skipped} failed={failed} "
                    f"path={relative_path}",
                    flush=True,
                )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    print(f"Complete: generated={generated} skipped={skipped} failed={failed} output={output_root}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
