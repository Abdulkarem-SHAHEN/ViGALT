#!/usr/bin/env python3
"""
Build a consolidated per-image evaluation dataset from raw Dataset/ folders.

Reads Dataset/N/ (image, DOM, alt texts, claim relevancy, evaluation JSONs)
and writes evaluation_dataset/N/ with alt_text_and_claims.txt (alt text +
plain claims, no relevancy labels), evaluation.json, plus a top-level
summary.json with averaged metrics.

Builds consolidated ViGALT vs CAID evaluation records from raw Dataset/ folders.

Usage:
  python -m viglat.build_dataset
  python -m viglat.build_dataset --source Dataset --output evaluation_dataset
  python -m viglat.build_dataset --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..evaluation.dataset_summary import SUMMARY_METRICS, build_main_dataset_summary
from ..evaluation.methods import METHOD_CAID, METHOD_VIGALT, resolve_method_key
from ..evaluation.utils import IMAGE_EXTENSIONS, build_relevancy_block
from ..paths import resolve_path

DEFAULT_SOURCE = "Dataset"
DEFAULT_OUTPUT = "evaluation_dataset"

OUTPUT_METHODS: tuple[str, ...] = (METHOD_VIGALT, METHOD_CAID)

ALT_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    METHOD_VIGALT: ("ViGALT_{n}.txt", "AAAA_{n}.txt"),
    METHOD_CAID: ("CAID_{n}.txt", "ASSETS24_{n}.txt"),
}

CLAIM_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    METHOD_VIGALT: ("CLAIM_RELEVANCY_ViGALT_{n}.txt", "CLAIM_RELEVANCY_AAAA_{n}.txt"),
    METHOD_CAID: ("CLAIM_RELEVANCY_CAID_{n}.txt", "CLAIM_RELEVANCY_ASSETS_{n}.txt"),
}


def first_existing_file(folder: Path, templates: tuple[str, ...], folder_number: int) -> Path:
    for template in templates:
        candidate = folder / template.format(n=folder_number)
        if candidate.exists():
            return candidate
    names = ", ".join(template.format(n=folder_number) for template in templates)
    raise FileNotFoundError(f"None of [{names}] found in {folder}")


def json_section(data: dict, method: str) -> dict:
    key = resolve_method_key(data, method)
    return data[key]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build consolidated evaluation_dataset/ from raw Dataset/ folders."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Source directory with numbered subfolders (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory contents.",
    )
    return parser.parse_args()


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def discover_folders(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    folders = [path for path in source.iterdir() if path.is_dir() and path.name.isdigit()]
    if not folders:
        raise FileNotFoundError(f"No numbered subfolders found in: {source}")

    return sorted(folders, key=lambda path: int(path.name))


def find_image(folder: Path) -> Path:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    raise FileNotFoundError(f"No image file found in {folder}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def parse_claims(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")

    claims: list[dict] = []
    for index, raw_line in enumerate(read_text(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "|" not in line:
            raise ValueError(f"Invalid claim line in {path}: {line!r}")

        claim_text, label = line.rsplit("|", 1)
        claim_text = claim_text.strip()
        label = label.strip()
        if not claim_text or not label:
            raise ValueError(f"Invalid claim line in {path}: {line!r}")

        claims.append(
            {
                "claim_id": f"C{index}",
                "claim": claim_text,
                "label": label,
            }
        )
    return claims


def format_alt_text_and_claims(
    alt_texts: dict[str, str],
    claims_by_method: dict[str, list[dict]],
) -> str:
    sections: list[str] = []
    for method in OUTPUT_METHODS:
        alt_text = alt_texts[method]
        claim_lines = [claim["claim"] for claim in claims_by_method[method]]
        body = alt_text
        if claim_lines:
            body = f"{alt_text}\n\n" + "\n".join(claim_lines)
        sections.append(f"=== {method} ===\n{body}")
    return "\n\n".join(sections) + "\n"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_method_eval(
    *,
    source_key: str,
    output_key: str,
    alt_text: str,
    claims: list[dict],
    efficiency_data: dict,
    objectivity_data: dict,
    redundancy_data: dict,
) -> dict:
    relevancy = build_relevancy_block(claims)
    redundancy_metrics = redundancy_data.get("metrics", {})
    objectivity_claims = objectivity_data.get("llm_output", {}).get("claims", [])
    redundancy_claims = redundancy_data.get("llm_output", {}).get("claims", [])

    return {
        "alt_text": alt_text,
        "relevancy": relevancy,
        "redundancy": {
            "metrics": redundancy_metrics,
            "input_claims": redundancy_data.get("input_claims", []),
            "claims": redundancy_claims,
        },
        "objectivity": {
            "objectivity_score": objectivity_data.get("objectivity_score"),
            "objective_claims": objectivity_data.get("objective_claims"),
            "subjective_claims": objectivity_data.get("subjective_claims"),
            "total_claims": objectivity_data.get("total_claims"),
            "claims": objectivity_claims,
        },
        "efficiency": {
            "word_count": efficiency_data.get("word_count"),
            "relevant_novel_claims": efficiency_data.get("relevant_novel_claims"),
            "efficiency_per_100_words": efficiency_data.get("efficiency_per_100_words"),
        },
        "_source_method_key": source_key,
        "_output_method_key": output_key,
    }


def build_folder_evaluation(
    *,
    folder_number: int,
    image_file: str,
    alt_texts: dict[str, str],
    claims_by_method: dict[str, list[dict]],
    efficiency_json: dict,
    objectivity_json: dict,
    redundancy_json: dict,
) -> dict:
    methods: dict[str, dict] = {}

    for method in OUTPUT_METHODS:
        methods[method] = build_method_eval(
            source_key=method,
            output_key=method,
            alt_text=alt_texts[method],
            claims=claims_by_method[method],
            efficiency_data=json_section(efficiency_json["results"], method),
            objectivity_data=json_section(objectivity_json, method),
            redundancy_data=json_section(redundancy_json, method),
        )
        methods[method].pop("_source_method_key", None)
        methods[method].pop("_output_method_key", None)

    return {
        "folder_number": folder_number,
        "image_file": image_file,
        "methods": methods,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_folder(source_folder: Path, output_folder: Path) -> dict:
    folder_number = int(source_folder.name)

    image_path = find_image(source_folder)
    dom_source = source_folder / f"DOM_JSON_{folder_number}.txt"
    efficiency_path = source_folder / f"efficiency_results_{folder_number}.json"
    objectivity_path = source_folder / f"objectivity_results_{folder_number}.json"
    redundancy_path = source_folder / f"redundancy_results_{folder_number}.json"

    alt_texts: dict[str, str] = {}
    claims_by_method: dict[str, list[dict]] = {}

    for method in OUTPUT_METHODS:
        alt_path = first_existing_file(source_folder, ALT_FILE_CANDIDATES[method], folder_number)
        claim_path = first_existing_file(source_folder, CLAIM_FILE_CANDIDATES[method], folder_number)
        alt_texts[method] = read_text(alt_path)
        claims_by_method[method] = parse_claims(claim_path)

    efficiency_json = load_json(efficiency_path)
    objectivity_json = load_json(objectivity_path)
    redundancy_json = load_json(redundancy_path)

    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, output_folder / image_path.name)
    shutil.copy2(dom_source, output_folder / "dom.json")

    alt_text_and_claims_path = output_folder / "alt_text_and_claims.txt"
    evaluation_path = output_folder / "evaluation.json"

    alt_text_and_claims_path.write_text(
        format_alt_text_and_claims(alt_texts, claims_by_method),
        encoding="utf-8",
    )

    evaluation = build_folder_evaluation(
        folder_number=folder_number,
        image_file=image_path.name,
        alt_texts=alt_texts,
        claims_by_method=claims_by_method,
        efficiency_json=efficiency_json,
        objectivity_json=objectivity_json,
        redundancy_json=redundancy_json,
    )
    write_json(evaluation_path, evaluation)
    return evaluation


def prepare_output_dir(output_root: Path, force: bool) -> None:
    if output_root.exists() and force:
        for child in output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    configure_stdout()

    source_root = resolve_path(args.source)
    output_root = resolve_path(args.output)
    prepare_output_dir(output_root, args.force)

    folders = discover_folders(source_root)
    evaluations: list[dict] = []

    print("=" * 60)
    print("Build Consolidated Evaluation Dataset")
    print("=" * 60)
    print(f"Source:  {source_root}")
    print(f"Output:  {output_root}")
    print(f"Folders: {len(folders)}")
    print()

    for source_folder in folders:
        folder_number = source_folder.name
        output_folder = output_root / folder_number
        evaluation = process_folder(source_folder, output_folder)
        evaluations.append(evaluation)
        print(f"[{folder_number}] wrote {output_folder.name}/")

    summary = build_main_dataset_summary(evaluations)
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Folders: {summary['folders_counted']}")
    for method, metrics in summary["methods"].items():
        print(f"  {method}:")
        print(f"    avg_relevancy_score:         {metrics['avg_relevancy_score']:.4f}")
        print(f"    avg_objectivity_score:       {metrics['avg_objectivity_score']:.4f}")
        print(f"    avg_efficiency_per_100_words:{metrics['avg_efficiency_per_100_words']:.4f}")
        print(f"    avg_novelty_rate:            {metrics['avg_novelty_rate']:.4f}")
        print(f"    avg_redundancy_rate:         {metrics['avg_redundancy_rate']:.4f}")
    print(f"Saved: {summary_path}")

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
