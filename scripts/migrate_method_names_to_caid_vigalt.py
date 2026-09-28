#!/usr/bin/env python3
"""One-shot idempotent rename: our/asset24/ASSETS24/AAAA -> ViGALT/CAID in data files."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", ".cursor", "node_modules"}

METHOD_KEY_RENAMES = {
    "ASSETS24": "CAID",
    "AAAA": "ViGALT",
}

ALGORITHM_RENAMES = {
    "our": "ViGALT",
    "asset24": "CAID",
}

ALT_TEXT_FIELD_RENAMES = {
    "asset24_alt_text": "caid_alt_text",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def transform_json_value(value, *, in_fusion_context: bool = False):
    if isinstance(value, dict):
        new_dict: dict = {}
        for key, item in value.items():
            new_key = METHOD_KEY_RENAMES.get(key, key)
            child_fusion = in_fusion_context or key in {"fusion", "parsed", "llm_output"}
            new_dict[new_key] = transform_json_value(item, in_fusion_context=child_fusion)
        if new_dict.get("algorithm") in ALGORITHM_RENAMES:
            algo = new_dict["algorithm"]
            new_dict["algorithm"] = ALGORITHM_RENAMES[algo]
            if new_dict["algorithm"] == "ViGALT":
                alt = new_dict.get("final_alt_text") or new_dict.get("vigalt_alt_text")
                if alt and "vigalt_alt_text" not in new_dict:
                    new_dict["vigalt_alt_text"] = alt
            if new_dict["algorithm"] == "CAID":
                for old_field, new_field in ALT_TEXT_FIELD_RENAMES.items():
                    if old_field in new_dict and new_field not in new_dict:
                        new_dict[new_field] = new_dict.pop(old_field)
        for old_field, new_field in ALT_TEXT_FIELD_RENAMES.items():
            if old_field in new_dict and new_field not in new_dict:
                new_dict[new_field] = new_dict.pop(old_field)
        return new_dict
    if isinstance(value, list):
        return [transform_json_value(item, in_fusion_context=in_fusion_context) for item in value]
    if isinstance(value, str) and value in ALGORITHM_RENAMES:
        return ALGORITHM_RENAMES[value]
    return value


def migrate_json_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
        data = json.loads(original)
    except (json.JSONDecodeError, OSError):
        return False
    transformed = transform_json_value(data)
    new_text = json.dumps(transformed, indent=2, ensure_ascii=False) + "\n"
    if new_text != original and not (original.endswith("\n") and new_text == original + "\n"):
        if new_text.rstrip("\n") == original.rstrip("\n"):
            return False
    if new_text != original:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def migrate_text_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    updated = text
    updated = updated.replace("=== ASSETS24 ===", "=== CAID ===")
    updated = updated.replace("=== AAAA ===", "=== ViGALT ===")
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def rename_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return True


def migrate_dataset_files(dataset_root: Path) -> list[str]:
    actions: list[str] = []
    if not dataset_root.is_dir():
        return actions
    for folder in dataset_root.iterdir():
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        n = folder.name
        renames = [
            (folder / f"ASSETS24_{n}.txt", folder / f"CAID_{n}.txt"),
            (folder / f"CLAIM_RELEVANCY_ASSETS_{n}.txt", folder / f"CLAIM_RELEVANCY_CAID_{n}.txt"),
            (folder / f"AAAA_{n}.txt", folder / f"ViGALT_{n}.txt"),
            (folder / f"CLAIM_RELEVANCY_AAAA_{n}.txt", folder / f"CLAIM_RELEVANCY_ViGALT_{n}.txt"),
        ]
        for src, dst in renames:
            if rename_if_exists(src, dst):
                actions.append(f"renamed {src.relative_to(ROOT)} -> {dst.name}")
    return actions


def migrate_output_dirs(output_root: Path) -> list[str]:
    actions: list[str] = []
    legacy = output_root / "ASSEST24_claim_lists"
    target = output_root / "CAID_claim_lists"
    if legacy.is_dir() and not target.exists():
        legacy.rename(target)
        actions.append(f"renamed {legacy.relative_to(ROOT)} -> CAID_claim_lists")
    return actions


def walk_and_migrate(root: Path) -> tuple[int, int]:
    json_changed = 0
    text_changed = 0
    for path in root.rglob("*"):
        if should_skip(path) or not path.is_file():
            continue
        if path.suffix.lower() == ".json":
            if migrate_json_file(path):
                json_changed += 1
        elif path.name == "alt_text_and_claims.txt":
            if migrate_text_file(path):
                text_changed += 1
    return json_changed, text_changed


def main() -> int:
    json_n, text_n = walk_and_migrate(ROOT)
    actions = migrate_dataset_files(ROOT / "Dataset")
    actions.extend(migrate_output_dirs(ROOT / "output"))
    print(f"Migrated JSON files: {json_n}")
    print(f"Migrated alt_text_and_claims.txt: {text_n}")
    for action in actions:
        print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
