#!/usr/bin/env python3
"""
Combine claim lists from both ALT algorithms with relevancy evaluations
into one text file per product/algorithm pair.

Sources:
  ViGALT — output/ViGALT_claim_lists/*_claims.json
  CAID   — output/CAID_claim_lists/*_claims.json
  relevancy — output/relevancy_evaluations.json

Usage:
  python -m viglat.combine_claims_relevancy
  python -m viglat.combine_claims_relevancy --output output/claims_with_labels
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..evaluation.methods import (
    DEFAULT_CAID_CLAIMS_DIR,
    DEFAULT_VIGALT_CLAIMS_DIR,
    METHOD_CAID,
    METHOD_VIGALT,
    normalize_method_id,
)
from ..paths import resolve_path

DEFAULT_RELEVANCY_FILE = "output/relevancy_evaluations.json"
DEFAULT_OUTPUT_DIR = "output/claims_with_labels"

ALGORITHM_ORDER = {METHOD_VIGALT: 0, METHOD_CAID: 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export claim labels into one minimal text file per product/algorithm."
    )
    parser.add_argument(
        "--vigalt-claims-dir",
        default=DEFAULT_VIGALT_CLAIMS_DIR,
        help=f"Directory with ViGALT claim lists (default: {DEFAULT_VIGALT_CLAIMS_DIR}).",
    )
    parser.add_argument(
        "--caid-claims-dir",
        default=DEFAULT_CAID_CLAIMS_DIR,
        help=f"Directory with CAID claim lists (default: {DEFAULT_CAID_CLAIMS_DIR}).",
    )
    parser.add_argument(
        "--relevancy",
        default=DEFAULT_RELEVANCY_FILE,
        help=f"Relevancy evaluations JSON file (default: {DEFAULT_RELEVANCY_FILE}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for text files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def product_index(product_stem: str) -> int:
    match = re.match(r"^(\d+)", product_stem)
    if not match:
        raise ValueError(f"Cannot extract product index from stem: {product_stem!r}")
    return int(match.group(1))


def load_claim_files(claims_dir: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    if not claims_dir.exists():
        return lookup

    for claim_file in sorted(claims_dir.glob("*_claims.json")):
        record = json.loads(claim_file.read_text(encoding="utf-8"))
        lookup[claim_file.name] = record
    return lookup


def source_claims_for_record(claim_record: dict | None) -> list[dict]:
    if not claim_record:
        return []
    parsed = claim_record.get("parsed")
    if not isinstance(parsed, dict):
        return []
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def merge_claims(
    source_claims: list[dict],
    evaluated_claims: list[dict],
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    eval_by_id = {
        str(claim.get("claim_id")): claim
        for claim in evaluated_claims
        if isinstance(claim, dict) and claim.get("claim_id")
    }

    merged: list[dict] = []
    for source in source_claims:
        claim_id = str(source.get("claim_id", ""))
        evaluation = eval_by_id.get(claim_id)
        if evaluation is None:
            warnings.append(f"Missing relevancy evaluation for claim_id {claim_id!r}.")
            merged.append(
                {
                    "claim": str(source.get("claim", "")),
                    "label": "",
                }
            )
            continue

        merged.append(
            {
                "claim": str(source.get("claim") or evaluation.get("claim") or ""),
                "label": str(evaluation.get("label") or ""),
            }
        )

    source_ids = {str(claim.get("claim_id")) for claim in source_claims if claim.get("claim_id")}
    for claim_id, evaluation in eval_by_id.items():
        if claim_id not in source_ids:
            warnings.append(f"Evaluation contains extra claim_id {claim_id!r} not in source file.")
            merged.append(
                {
                    "claim": str(evaluation.get("claim", "")),
                    "label": str(evaluation.get("label") or ""),
                }
            )

    return merged, warnings


def build_blocks(
    results: list[dict],
    vigalt_claims: dict[str, dict],
    caid_claims: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    blocks: list[dict] = []

    for result in results:
        if not isinstance(result, dict):
            continue

        algorithm = normalize_method_id(str(result.get("algorithm", "")))
        claim_list_file = Path(str(result.get("claim_list_file", "")))
        claim_filename = claim_list_file.name
        if algorithm == METHOD_VIGALT:
            claim_record = vigalt_claims.get(claim_filename)
        elif algorithm == METHOD_CAID:
            claim_record = caid_claims.get(claim_filename)
        else:
            claim_record = None

        if claim_record is None:
            warnings.append(f"No source claim file found for {algorithm}:{claim_filename}.")

        evaluation = result.get("evaluation")
        if not isinstance(evaluation, dict):
            warnings.append(f"Missing evaluation for {algorithm}:{claim_filename}.")
            continue

        if result.get("status") != "ok":
            warnings.append(
                f"Skipped non-ok evaluation for {algorithm}:{claim_filename} (status={result.get('status')!r})."
            )
            continue

        evaluated_claims = evaluation.get("claims")
        if not isinstance(evaluated_claims, list):
            warnings.append(f"Missing evaluated claims for {algorithm}:{claim_filename}.")
            continue

        source_claims = source_claims_for_record(claim_record)
        if not source_claims:
            source_claims = result.get("input_claims") if isinstance(result.get("input_claims"), list) else []

        merged_claims, claim_warnings = merge_claims(source_claims, evaluated_claims)
        for warning in claim_warnings:
            warnings.append(f"{algorithm}:{claim_filename}: {warning}")

        product_stem = str(result.get("product_stem") or (claim_record or {}).get("product_stem") or "")

        blocks.append(
            {
                "algorithm": algorithm,
                "product_stem": product_stem,
                "product_index": product_index(product_stem) if product_stem else 0,
                "claims": merged_claims,
                "claim_filename": claim_filename,
            }
        )

    blocks.sort(
        key=lambda block: (
            block["product_index"],
            ALGORITHM_ORDER.get(block["algorithm"], 99),
        )
    )
    return blocks, warnings


def block_output_path(output_dir: Path, block: dict) -> Path:
    return output_dir / f"{block['product_stem']}_{block['algorithm']}.txt"


def format_block_file(claims: list[dict]) -> str:
    lines = [f"{claim['claim']} | {claim['label']}" for claim in claims]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def write_block_files(output_dir: Path, blocks: list[dict]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for block in blocks:
        path = block_output_path(output_dir, block)
        path.write_text(format_block_file(block["claims"]), encoding="utf-8")
        written.append(path)

    return written


def find_unmatched_claim_files(
    vigalt_claims: dict[str, dict],
    caid_claims: dict[str, dict],
    blocks: list[dict],
) -> list[str]:
    warnings: list[str] = []
    matched = {(block["algorithm"], block["claim_filename"]) for block in blocks}

    for filename in vigalt_claims:
        if (METHOD_VIGALT, filename) not in matched:
            warnings.append(f"No relevancy result found for ViGALT claim file {filename}.")

    for filename in caid_claims:
        if (METHOD_CAID, filename) not in matched:
            warnings.append(f"No relevancy result found for CAID claim file {filename}.")

    return warnings


def main() -> int:
    configure_stdout()
    args = parse_args()

    vigalt_claims_dir = resolve_path(args.vigalt_claims_dir)
    caid_claims_dir = resolve_path(args.caid_claims_dir)
    relevancy_path = resolve_path(args.relevancy)
    output_dir = resolve_path(args.output)

    if not relevancy_path.exists():
        print(f"Relevancy file not found: {relevancy_path}", file=sys.stderr)
        return 1

    relevancy_data = json.loads(relevancy_path.read_text(encoding="utf-8"))
    results = relevancy_data.get("results")
    if not isinstance(results, list):
        print(f"Invalid relevancy file (missing results list): {relevancy_path}", file=sys.stderr)
        return 1

    vigalt_claims = load_claim_files(vigalt_claims_dir)
    caid_claims = load_claim_files(caid_claims_dir)

    blocks, warnings = build_blocks(results, vigalt_claims, caid_claims)
    warnings.extend(find_unmatched_claim_files(vigalt_claims, caid_claims, blocks))

    written = write_block_files(output_dir, blocks)

    print(f"Wrote {len(written)} files to {output_dir}")
    if warnings:
        print(f"Warnings: {len(warnings)}", file=sys.stderr)
        for warning in warnings:
            print(f"- {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
