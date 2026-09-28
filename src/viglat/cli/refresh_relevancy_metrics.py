#!/usr/bin/env python3
"""
Recompute relevancy blocks and dataset summaries from existing evaluation.json files.

Uses the weighted relevancy formula from prompts/evaluators/relevancy.txt without
re-running LLM evaluators.

Usage:
  python -m viglat.refresh_relevancy_metrics
  python -m viglat.refresh_relevancy_metrics --dataset-dir evaluation_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..evaluation.ablation import ABLATION_VARIANTS, build_ablations_summary
from ..evaluation.dataset_summary import build_main_dataset_summary
from ..evaluation.utils import build_relevancy_block, configure_stdout, save_output
from ..paths import resolve_path

DEFAULT_DATASET_DIR = "evaluation_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh relevancy_score fields and summary JSON from evaluation.json files."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Root evaluation dataset directory (default: {DEFAULT_DATASET_DIR}).",
    )
    return parser.parse_args()


def discover_evaluation_paths(dataset_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for folder in sorted(dataset_dir.iterdir(), key=lambda p: p.name):
        if folder.is_dir() and folder.name.isdigit():
            eval_path = folder / "evaluation.json"
            if eval_path.exists():
                paths.append(eval_path)
            for variant in ABLATION_VARIANTS:
                ablation_path = folder / "ablations" / variant / "evaluation.json"
                if ablation_path.exists():
                    paths.append(ablation_path)
    return paths


def refresh_relevancy_in_evaluation(evaluation: dict) -> bool:
    methods = evaluation.get("methods")
    if not isinstance(methods, dict):
        return False

    changed = False
    for method_eval in methods.values():
        if not isinstance(method_eval, dict):
            continue
        relevancy = method_eval.get("relevancy")
        if not isinstance(relevancy, dict):
            continue
        claims = relevancy.get("claims")
        if not isinstance(claims, list):
            continue

        updated = build_relevancy_block(claims)
        if relevancy.pop("relevancy_rate", None) is not None:
            changed = True
        for key, value in updated.items():
            if relevancy.get(key) != value:
                relevancy[key] = value
                changed = True
    return changed


def load_main_evaluations(dataset_dir: Path) -> list[dict]:
    folders = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    evaluations: list[dict] = []
    for folder in folders:
        eval_path = folder / "evaluation.json"
        if eval_path.exists():
            evaluations.append(json.loads(eval_path.read_text(encoding="utf-8")))
    if not evaluations:
        raise FileNotFoundError(f"No main evaluation.json files under {dataset_dir}")
    return evaluations


def rebuild_ablations_summary(dataset_dir: Path) -> dict:
    results: list[dict] = []
    for product_dir in sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        product_id = int(product_dir.name)
        for variant in ABLATION_VARIANTS:
            output_dir = product_dir / "ablations" / variant
            evaluation_path = output_dir / "evaluation.json"
            if evaluation_path.exists():
                results.append(
                    {
                        "status": "ok",
                        "variant": variant,
                        "product_id": product_id,
                        "output_dir": str(output_dir),
                    }
                )
    return build_ablations_summary(results)


def run(args: argparse.Namespace) -> int:
    configure_stdout()
    dataset_dir = resolve_path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    eval_paths = discover_evaluation_paths(dataset_dir)
    updated_files = 0
    for eval_path in eval_paths:
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        if refresh_relevancy_in_evaluation(evaluation):
            eval_path.write_text(
                json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            updated_files += 1

    main_evaluations = load_main_evaluations(dataset_dir)
    summary = build_main_dataset_summary(main_evaluations)
    summary_path = dataset_dir / "summary.json"
    save_output(summary_path, summary)

    ablations_summary = rebuild_ablations_summary(dataset_dir)
    save_output(dataset_dir / "ablations_summary.json", ablations_summary)

    print(f"Refreshed relevancy in {updated_files} evaluation.json file(s).")
    print(f"Wrote {summary_path}")
    print(f"Wrote {dataset_dir / 'ablations_summary.json'}")
    for method, metrics in summary["methods"].items():
        score = metrics.get("avg_relevancy_score")
        if score is not None:
            print(f"  {method} avg_relevancy_score: {score:.4f} ({score * 100:.2f}%)")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
