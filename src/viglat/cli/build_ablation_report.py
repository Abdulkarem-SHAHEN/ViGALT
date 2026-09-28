#!/usr/bin/env python3
"""
Build fusion-stage ablation summary JSON and markdown report from existing artifacts.

Usage:
  python -m viglat.build_ablation_report
  python -m viglat.build_ablation_report --dataset-dir evaluation_dataset --report-path docs/validator_ablation_report.md
"""

from __future__ import annotations

import argparse
import sys

from ..evaluation.ablation import parse_product_ids
from ..evaluation.ablation_report import (
    DEFAULT_DATASET_DIR,
    DEFAULT_REPORT_NAME,
    DEFAULT_SUMMARY_NAME,
    discover_product_ids,
    write_ablation_report,
)
from ..evaluation.utils import configure_stdout
from ..paths import project_root, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate validator ablation results and write ablation-only markdown section."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Evaluation dataset root (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--products",
        default=None,
        help="Optional product IDs (e.g. '1-30'); default: all completed ablation folders.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help=f"Output JSON path (default: <dataset-dir>/{DEFAULT_SUMMARY_NAME}).",
    )
    parser.add_argument(
        "--report-path",
        default=f"docs/{DEFAULT_REPORT_NAME}",
        help=f"Output markdown path (default: docs/{DEFAULT_REPORT_NAME}).",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    dataset_dir = resolve_path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Error: dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    if args.products:
        try:
            product_ids = parse_product_ids(args.products)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        product_ids = discover_product_ids(dataset_dir)
        if not product_ids:
            print(
                f"Error: no completed ablation products found under {dataset_dir}",
                file=sys.stderr,
            )
            return 1

    summary_path = resolve_path(args.summary_path) if args.summary_path else dataset_dir / DEFAULT_SUMMARY_NAME
    report_path = resolve_path(args.report_path)

    try:
        summary = write_ablation_report(
            dataset_dir=dataset_dir,
            summary_path=summary_path,
            report_path=report_path,
            product_ids=product_ids,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Validator Ablation Report")
    print("=" * 60)
    print(f"Products analyzed: {summary['products_analyzed']}")
    print(f"Summary JSON:      {summary_path.relative_to(project_root())}")
    print(f"Markdown report:   {report_path.relative_to(project_root())}")
    print()
    print("Primary outcomes:")
    comp = summary["completeness"]
    acc = summary["accuracy"]
    red = summary["redundancy"]
    print(
        f"  Completeness: {comp['added_lost_in_no_completeness']}/{comp['total_added_claims']} ADDED claims lost; "
        f"avg Δ claims {comp['avg_delta_claims_no_completeness']:.2f}, "
        f"avg Δ words {comp['avg_delta_words_no_completeness']:.2f}"
    )
    print(
        f"  Accuracy:     {acc['products_with_interventions']}/{summary['products_analyzed']} products with "
        f"CORRECTED/DELETED ({acc['total_corrected']} corrected, {acc['total_deleted']} deleted)"
    )
    print(
        f"  Redundancy:   avg Δ avoidable rate {red['avg_delta_avoidable_rate_no_redundancy']:.3f}; "
        f"{red['products_increased_avoidable_rate']} products increased avoidable redundancy"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
