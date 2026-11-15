"""Guards the docs against drifting from the scripts: every documented option must exist in the script it is used with."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "docs").glob("*.md"))]
FLAG_PATTERN = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
# Options of external programs that the docs show next to this project's commands.
EXTERNAL_FLAGS = {"--index-url", "--port", "--listen", "--help"}
FORWARDING_SCRIPTS = {"run_batch.sh": ["qwen_resourcepack_batch.py"]}


def script_flags(name: str) -> set[str]:
    flags = set(FLAG_PATTERN.findall((PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")))
    for forwarded in FORWARDING_SCRIPTS.get(name, []):
        flags |= script_flags(forwarded)
    return flags


def all_script_flags() -> set[str]:
    names = [path.name for path in (PROJECT_ROOT / "scripts").iterdir() if path.suffix in {".py", ".sh"}]
    return set().union(*(script_flags(name) for name in names)) | EXTERNAL_FLAGS


def documented_commands() -> list[tuple[Path, str]]:
    commands: list[tuple[Path, str]] = []
    for doc in DOC_FILES:
        in_fence = False
        pending = ""
        for line in doc.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                pending = ""
                continue
            if not in_fence:
                continue
            pending += " " + line.strip().removesuffix("\\")
            if not line.rstrip().endswith("\\"):
                commands.append((doc, pending.strip()))
                pending = ""
    return commands


def test_every_flag_in_the_docs_exists_in_some_script() -> None:
    known = all_script_flags()
    unknown = {
        (doc.name, flag)
        for doc in DOC_FILES
        for flag in FLAG_PATTERN.findall(doc.read_text(encoding="utf-8"))
        if flag not in known
    }

    assert not unknown, f"Documented options that no script defines: {sorted(unknown)}"


def test_flags_in_documented_commands_belong_to_the_script_they_are_used_with() -> None:
    problems = []
    for doc, command in documented_commands():
        match = re.search(r"scripts/([\w.]+\.(?:py|sh))", command)
        if match is None:
            continue
        allowed = script_flags(match.group(1)) | EXTERNAL_FLAGS
        problems += [(doc.name, match.group(1), flag) for flag in FLAG_PATTERN.findall(command) if flag not in allowed]

    assert not problems, f"Flags used with the wrong script: {problems}"


def test_every_script_documented_in_a_command_exists() -> None:
    missing = {
        match
        for _, command in documented_commands()
        for match in re.findall(r"scripts/([\w.]+\.(?:py|sh))", command)
        if not (PROJECT_ROOT / "scripts" / match).is_file()
    }

    assert not missing, f"Documented scripts that do not exist: {sorted(missing)}"
