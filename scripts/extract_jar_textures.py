#!/usr/bin/env python3
"""Extract textures (and their animation metadata) from the JAR files in a directory.

Works for mod JARs and for a vanilla client JAR alike. Files are written below the output directory without the
leading ``assets/`` segment, so ``--output-dir mods/assets`` yields ``mods/assets/<namespace>/textures/...``.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

TEXTURE_SUFFIXES = (".png", ".png.mcmeta")


@dataclass(frozen=True)
class ExtractionResult:
    """Counts of JARs that yielded textures, had none or were unreadable, and that failed."""

    extracted: int
    skipped: int
    errors: int


def is_texture_entry(name: str) -> bool:
    """True for ``assets/...`` PNG textures and their ``.mcmeta`` files, rejecting path traversal."""
    parts = Path(name).parts
    return (
        len(parts) > 1
        and parts[0] == "assets"
        and ".." not in parts
        and not name.endswith("/")
        and name.endswith(TEXTURE_SUFFIXES)
    )


def extract_jar_textures(jars_dir: Path, output_dir: Path) -> ExtractionResult:
    """Extract textures from every ``*.jar`` in ``jars_dir`` into ``output_dir``, never overwriting existing files."""
    extracted = skipped = errors = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    jars = sorted(jars_dir.glob("*.jar"))
    print(f"Found {len(jars)} JARs in {jars_dir}")
    for jar in jars:
        try:
            with zipfile.ZipFile(jar, "r") as archive:
                entries = [name for name in archive.namelist() if is_texture_entry(name)]
                if not entries:
                    skipped += 1
                    continue
                for entry in entries:
                    destination = output_dir / Path(*Path(entry).parts[1:])
                    if destination.exists():
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(entry))
                extracted += 1
                print(f"  [OK] {jar.name} ({len(entries)} files)")
        except zipfile.BadZipFile:
            print(f"  [SKIP] {jar.name} (not a valid zip)", file=sys.stderr)
            skipped += 1
        except OSError as error:
            print(f"  [ERR] {jar.name}: {error}", file=sys.stderr)
            errors += 1
    return ExtractionResult(extracted, skipped, errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jars-dir", type=Path, required=True, help="Directory containing the JAR files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mods/assets"),
        help="Destination for the extracted assets (default: mods/assets).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = extract_jar_textures(args.jars_dir, args.output_dir)
    print(f"\nDone: {result.extracted} JARs extracted, {result.skipped} skipped, {result.errors} errors")
    print(f"Output: {args.output_dir.resolve()}")
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
