#!/usr/bin/env python3
"""
Run validator ablation study: fuse without one validator, then evaluate.

For each product with pipeline/ artifacts under evaluation_dataset/, runs 3 fusion ablations (no_accuracy,
no_completeness, no_redundancy) with Luna, then evaluates each result with
Relevancy, Redundancy, Objectivity, and Efficiency using Terra.

Usage:
  python -m viglat.run_validator_ablation
  python -m viglat.run_validator_ablation --skip-existing --delay 3
  python -m viglat.run_validator_ablation --products 1-5 --variants no_accuracy
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI

from ..backend.config import DEFAULT_EVALUATION_MODEL, DEFAULT_OPENAI_MODEL
from ..evaluation.ablation import (
    DEFAULT_DATASET_DIR,
    parse_product_ids,
    parse_variants,
    run_ablation_batch,
)
from ..evaluation.utils import configure_stdout
from ..paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validator ablation fusion + evaluation batch study."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Evaluation dataset root (reads pipeline/, writes ablations/) (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--products",
        default=None,
        help="Product IDs to run, e.g. '1', '1,5,10', or '1-5' (default: all 30).",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated ablation variants (default: all three).",
    )
    parser.add_argument(
        "--fuser-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model for fusion ablations (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--claim-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model for post-fusion claim extraction (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--eval-model",
        default=DEFAULT_EVALUATION_MODEL,
        help=f"OpenAI model for evaluators (default: {DEFAULT_EVALUATION_MODEL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request OpenAI timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to wait between jobs (default: 3.0).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose evaluation.json already has status ok.",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        return 1

    try:
        product_ids = parse_product_ids(args.products)
        variants = parse_variants(args.variants)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    dataset_dir = resolve_path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Error: dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    summary = run_ablation_batch(
        dataset_dir=dataset_dir,
        product_ids=product_ids,
        variants=variants,
        fuser_model=args.fuser_model,
        claim_model=args.claim_model,
        eval_model=args.eval_model,
        timeout_s=float(args.timeout),
        delay_s=float(args.delay),
        skip_existing=args.skip_existing,
        client=client,
    )

    failed = summary.get("overall", {}).get("failed", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
