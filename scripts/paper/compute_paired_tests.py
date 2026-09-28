#!/usr/bin/env python3
"""Paired Wilcoxon tests; writes paired_statistical_tests.csv from evaluation_dataset/."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from viglat.evaluation.dataset_summary import metric_value  # noqa: E402

DEFAULT_DATASET = ROOT / "evaluation_dataset"
DEFAULT_OUTPUT_PATHS = (
    ROOT / "paired_statistical_tests.csv",
    ROOT / "ViGALT_short_4page" / "paired_statistical_tests.csv",
)

METRICS: tuple[tuple[str, str, bool], ...] = (
    ("Relevancy", "relevancy_score", True),
    ("Objectivity", "objectivity_score", True),
    ("Relevant Novelty", "novelty_rate", True),
    ("Avoidable Redundancy", "avoidable_redundancy_rate", False),
    ("Efficiency", "efficiency_per_100_words", True),
)


def load_paired_series(dataset_dir: Path, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    vigalt: list[float] = []
    caid: list[float] = []
    folders = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for folder in folders:
        evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        methods = evaluation["methods"]
        vigalt.append(float(metric_value(methods["ViGALT"], metric_key)))
        caid.append(float(metric_value(methods["CAID"], metric_key)))
    return np.array(vigalt), np.array(caid)


def bootstrap_mean_ci(differences: np.ndarray, *, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(differences)
    samples = np.array([rng.choice(differences, n, replace=True).mean() for _ in range(n_boot)])
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def rank_biserial(w_statistic: float, differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    n = len(nonzero)
    if n == 0:
        return float("nan")
    return float(1.0 - (2.0 * w_statistic) / (n * (n + 1) / 2.0))


def holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, (index, p_raw) in enumerate(indexed, start=1):
        corrected = min(1.0, (m - rank + 1) * p_raw)
        corrected = max(corrected, prev)
        prev = corrected
        adjusted[index] = corrected
    return adjusted


def analyze_metric(
    vigalt: np.ndarray,
    caid: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict[str, float | int | bool]:
    differences = vigalt - caid
    if higher_is_better:
        _, p_directional = stats.wilcoxon(vigalt, caid, alternative="greater", zero_method="wilcox")
    else:
        _, p_directional = stats.wilcoxon(vigalt, caid, alternative="less", zero_method="wilcox")

    w_stat, p_raw = stats.wilcoxon(vigalt, caid, zero_method="wilcox")
    ci_low, ci_high = bootstrap_mean_ci(differences)
    vigalt_higher = int((differences > 0).sum())
    caid_higher = int((differences < 0).sum())
    ties = int((differences == 0).sum())

    return {
        "vigalt_macro_mean": float(vigalt.mean()),
        "caid_macro_mean": float(caid.mean()),
        "paired_mean_difference_vigalt_minus_caid": float(differences.mean()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_W": float(w_stat),
        "p_raw": float(p_raw),
        "rank_biserial_r": rank_biserial(float(w_stat), differences),
        "vigalt_higher_images": vigalt_higher,
        "caid_higher_images": caid_higher,
        "ties": ties,
        "higher_is_better": higher_is_better,
        "p_directional": float(p_directional),
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "metric",
        "vigalt_macro_mean",
        "caid_macro_mean",
        "paired_mean_difference_vigalt_minus_caid",
        "ci95_low",
        "ci95_high",
        "wilcoxon_W",
        "p_raw",
        "rank_biserial_r",
        "vigalt_higher_images",
        "caid_higher_images",
        "ties",
        "higher_is_better",
        "p_holm",
        "significant_after_holm_0.05",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    if len(sys.argv) > 2:
        output_paths = (Path(sys.argv[2]),)
    else:
        output_paths = DEFAULT_OUTPUT_PATHS

    analyzed: list[dict] = []
    p_raw_values: list[float] = []
    for name, key, higher in METRICS:
        vigalt, caid = load_paired_series(dataset_dir, key)
        result = analyze_metric(vigalt, caid, higher_is_better=higher)
        result["metric"] = name
        analyzed.append(result)
        p_raw_values.append(float(result["p_raw"]))

    p_holm = holm_correction(p_raw_values)
    for row, corrected in zip(analyzed, p_holm):
        row.pop("p_directional", None)
        row["p_holm"] = corrected
        row["significant_after_holm_0.05"] = corrected < 0.05

    for output_path in output_paths:
        if not output_path.parent.exists():
            continue
        write_csv(analyzed, output_path)
        print(f"Wrote {output_path}")
    for row in analyzed:
        if row["metric"] == "Relevancy":
            print(
                f"  Relevancy: ViGALT {row['vigalt_macro_mean']*100:.2f}% vs "
                f"CAID {row['caid_macro_mean']*100:.2f}% "
                f"(delta={row['paired_mean_difference_vigalt_minus_caid']*100:.2f} pp, "
                f"p_Holm={row['p_holm']:.2e})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
