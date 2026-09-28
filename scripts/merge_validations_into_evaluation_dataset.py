#!/usr/bin/env python3
"""
Copy pre-fusion pipeline artifacts from ViGALT_validations/ into evaluation_dataset/N/pipeline/.

Copies text/JSON only (no images). Skips files that already exist under pipeline/.
Removes ViGALT_validations/ when all products are complete unless --no-remove-source.

Usage:
  python scripts/merge_validations_into_evaluation_dataset.py
  python scripts/merge_validations_into_evaluation_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PIPELINE_FILES = (
    "generated_alt_text.txt",
    "claims.json",
    "accuracy_validation.json",
    "completeness_validation.json",
    "redundancy_validation.json",
)

DEFAULT_SOURCE = "ViGALT_validations"
DEFAULT_DATASET = "evaluation_dataset"
MAX_PRODUCT = 30


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ViGALT_validations artifacts into evaluation_dataset/N/pipeline/."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Legacy validations root (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Evaluation dataset root (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying or deleting.",
    )
    parser.add_argument(
        "--no-remove-source",
        action="store_true",
        help="Do not delete the source validations directory after a successful merge.",
    )
    return parser.parse_args()


def merge_product(
    *,
    product_id: int,
    source_root: Path,
    dataset_root: Path,
    dry_run: bool,
) -> dict[str, int | list[str]]:
    pid = str(product_id)
    product_dir = dataset_root / pid
    eval_path = product_dir / "evaluation.json"
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing {eval_path}")

    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    image_file = evaluation.get("image_file")
    if isinstance(image_file, str) and image_file:
        image_path = product_dir / image_file
        if not image_path.is_file():
            print(f"  warning: evaluation.json image_file not found: {image_path}", file=sys.stderr)

    source_dir = source_root / pid
    pipeline_dir = product_dir / "pipeline"
    stats: dict[str, int | list[str]] = {
        "copied": 0,
        "skipped": 0,
        "missing_source": 0,
        "missing_names": [],
    }

    if not source_dir.is_dir():
        stats["missing_source"] = len(PIPELINE_FILES)
        stats["missing_names"] = list(PIPELINE_FILES)
        return stats

    if not dry_run:
        pipeline_dir.mkdir(parents=True, exist_ok=True)

    for name in PIPELINE_FILES:
        src = source_dir / name
        dest = pipeline_dir / name
        if dest.is_file():
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if not src.is_file():
            stats["missing_source"] = int(stats["missing_source"]) + 1
            missing = stats["missing_names"]
            assert isinstance(missing, list)
            missing.append(name)
            continue
        if dry_run:
            print(f"  would copy {src.relative_to(project_root())} -> {dest.relative_to(project_root())}")
        else:
            shutil.copy2(src, dest)
        stats["copied"] = int(stats["copied"]) + 1

    return stats


def main() -> int:
    args = parse_args()
    root = project_root()
    source_root = (root / args.source).resolve()
    dataset_root = (root / args.dataset).resolve()

    if not dataset_root.is_dir():
        print(f"Error: dataset not found: {dataset_root}", file=sys.stderr)
        return 1

    totals = {"copied": 0, "skipped": 0, "missing_source": 0}
    incomplete: list[int] = []

    print("=" * 60)
    print("Merge validations into evaluation_dataset")
    print("=" * 60)
    print(f"Source:  {source_root}")
    print(f"Dataset: {dataset_root}")
    if args.dry_run:
        print("Mode:    dry-run")
    print()

    for product_id in range(1, MAX_PRODUCT + 1):
        try:
            stats = merge_product(
                product_id=product_id,
                source_root=source_root,
                dataset_root=dataset_root,
                dry_run=args.dry_run,
            )
        except FileNotFoundError as exc:
            print(f"product {product_id}: ERROR {exc}", file=sys.stderr)
            incomplete.append(product_id)
            continue

        totals["copied"] += int(stats["copied"])
        totals["skipped"] += int(stats["skipped"])
        totals["missing_source"] += int(stats["missing_source"])
        missing_names = stats["missing_names"]
        assert isinstance(missing_names, list)
        if missing_names:
            incomplete.append(product_id)
            print(
                f"product {product_id}: incomplete — missing source: {', '.join(missing_names)}"
            )
        else:
            print(
                f"product {product_id}: copied={stats['copied']} skipped={stats['skipped']}"
            )

    print()
    print(
        f"Totals: copied={totals['copied']} skipped={totals['skipped']} "
        f"missing_source={totals['missing_source']}"
    )

    if incomplete:
        print(f"Incomplete products: {incomplete}", file=sys.stderr)
        return 1

    if args.dry_run:
        return 0

    if not args.no_remove_source and source_root.is_dir():
        shutil.rmtree(source_root)
        print(f"Removed source directory: {source_root.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
