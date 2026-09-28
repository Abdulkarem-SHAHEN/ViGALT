#!/usr/bin/env python3
"""
Generate publication figures for the ViGALT vs CAID paper evaluation.

Requires visualization extras:
  pip install -e ".[viz]"

Usage:
  python -m viglat.generate_paper_figures
  python -m viglat.generate_paper_figures --case-study-clothing 5 --case-study-furniture 24
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from ..evaluation.utils import configure_stdout
from ..paths import resolve_path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 3,
            "axes.linewidth": 1.5,
            "pdf.fonttype": 42,
        }
    )
except ImportError:
    plt = None  # type: ignore[assignment,misc]
    sns = None  # type: ignore[assignment,misc]

DEFAULT_DATASET_DIR = "evaluation_dataset"
DEFAULT_OUTPUT_DIR = "docs/figures"

METHOD_OUR = "ViGALT"
METHOD_BASELINE = "CAID"
DISPLAY_OUR = "Our Pipeline"
DISPLAY_BASELINE = "CAID"

COLOR_OUR = "#0173B2"
COLOR_BASELINE = "#949494"

CORE_METRICS = (
    ("avg_objectivity_score", "Objectivity"),
    ("avg_relevancy_score", "Relevancy"),
    ("avg_novelty_rate", "Relevant Novelty"),
    ("avg_avoidable_redundancy_rate", "Avoidable Redundancy"),
)

REDUNDANCY_SEGMENTS = (
    ("raw_novel_claims", "NOVEL", "#2CA02C"),
    ("unavoidable_repetition_claims", "UNAVOIDABLE_REPETITION", "#FFBF00"),
    ("avoidable_redundancy_claims", "AVOIDABLE_REDUNDANCY", "#E45756"),
)

CLOTHING_FOLDER_RANGE = (1, 15)
FURNITURE_FOLDER_RANGE = (16, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper evaluation figures from evaluation_dataset/."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Evaluation dataset root (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for figures (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=["pdf", "png"],
        help="Image formats to write for matplotlib figures (default: pdf png).",
    )
    parser.add_argument(
        "--case-study-clothing",
        type=int,
        default=None,
        help="Clothing product folder ID (1–15) for case study (default: auto-rank).",
    )
    parser.add_argument(
        "--case-study-furniture",
        type=int,
        default=None,
        help="Furniture product folder ID (16–30) for case study (default: auto-rank).",
    )
    parser.add_argument(
        "--skip-case-study",
        action="store_true",
        help="Skip LaTeX case study generation.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "core_metrics", "redundancy_stack", "efficiency_relevancy", "case_study"),
        default="all",
        help="Generate only one figure type (default: all).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_evaluations(dataset_dir: Path) -> list[dict]:
    folders = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    evaluations: list[dict] = []
    for folder in folders:
        eval_path = folder / "evaluation.json"
        if not eval_path.exists():
            continue
        evaluations.append(load_json(eval_path))
    if not evaluations:
        raise FileNotFoundError(f"No evaluation.json files under {dataset_dir}")
    return evaluations


def load_summary(dataset_dir: Path) -> dict:
    summary_path = dataset_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    return load_json(summary_path)


def method_display(method_key: str) -> str:
    if method_key == METHOD_OUR:
        return DISPLAY_OUR
    if method_key == METHOD_BASELINE:
        return DISPLAY_BASELINE
    return method_key


def pct(rate: float | None) -> float:
    if rate is None:
        raise ValueError("Missing rate for percentage conversion.")
    return rate * 100.0


def micro_redundancy_breakdown(evaluations: list[dict]) -> dict[str, dict[str, float]]:
    """Pool claim counts across products; segment shares sum to 100% per method."""
    totals: dict[str, dict[str, int]] = {
        METHOD_OUR: {key: 0 for key, _, _ in REDUNDANCY_SEGMENTS},
        METHOD_BASELINE: {key: 0 for key, _, _ in REDUNDANCY_SEGMENTS},
    }
    claim_totals: dict[str, int] = {METHOD_OUR: 0, METHOD_BASELINE: 0}

    for evaluation in evaluations:
        for method in (METHOD_OUR, METHOD_BASELINE):
            metrics = evaluation["methods"][method]["redundancy"]["metrics"]
            total = int(metrics["total_claims"])
            claim_totals[method] += total
            for count_key, _, _ in REDUNDANCY_SEGMENTS:
                totals[method][count_key] += int(metrics[count_key])

    breakdown: dict[str, dict[str, float]] = {}
    for method in (METHOD_OUR, METHOD_BASELINE):
        total = claim_totals[method]
        if total == 0:
            raise ValueError(f"No claims aggregated for method {method}")
        breakdown[method] = {
            count_key: (totals[method][count_key] / total) * 100.0
            for count_key, _, _ in REDUNDANCY_SEGMENTS
        }
    return breakdown


def require_matplotlib():
    if plt is None or sns is None:
        raise SystemExit(
            "matplotlib and seaborn are required. Install with: pip install -e \".[viz]\""
        )
    return plt, sns


def configure_plotting() -> None:
    _, sns_module = require_matplotlib()
    sns_module.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 3,
            "axes.linewidth": 1.5,
            "pdf.fonttype": 42,
        }
    )


class FigureWriter:
    def __init__(self, output_dir: Path, formats: list[str], plt) -> None:
        self.output_dir = output_dir
        self.formats = formats
        self.plt = plt

    def save(self, stem: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in self.formats:
            path = self.output_dir / f"{stem}.{fmt}"
            self.plt.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        self.plt.close()


def plot_core_metrics(summary: dict, writer: FigureWriter) -> None:
    plt = writer.plt
    our = summary["methods"][METHOD_OUR]
    baseline = summary["methods"][METHOD_BASELINE]

    labels = [label for _, label in CORE_METRICS]
    x = range(len(labels))
    width = 0.36

    our_vals = [pct(our[key]) for key, _ in CORE_METRICS]
    base_vals = [pct(baseline[key]) for key, _ in CORE_METRICS]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        [i - width / 2 for i in x],
        our_vals,
        width,
        label=DISPLAY_OUR,
        color=COLOR_OUR,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        base_vals,
        width,
        label=DISPLAY_BASELINE,
        color=COLOR_BASELINE,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylabel("Score (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("Core Evaluation Metrics")

    avoidable_idx = len(labels) - 1
    ax.annotate(
        "",
        xy=(avoidable_idx, 8),
        xytext=(avoidable_idx, 22),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    ax.text(
        avoidable_idx + 0.05,
        24,
        "Lower is better",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    writer.save("core_metrics")


def plot_redundancy_stack(breakdown: dict[str, dict[str, float]], writer: FigureWriter) -> None:
    plt = writer.plt
    methods = [METHOD_OUR, METHOD_BASELINE]
    x_labels = [method_display(m) for m in methods]
    x = range(len(methods))

    fig, ax = plt.subplots(figsize=(7, 6))
    bottoms = [0.0, 0.0]

    for count_key, segment_label, color in REDUNDANCY_SEGMENTS:
        heights = [breakdown[m][count_key] for m in methods]
        bars = ax.bar(x, heights, bottom=bottoms, label=segment_label.replace("_", " "), color=color, edgecolor="white", linewidth=0.6)
        for bar, height, bottom in zip(bars, heights, bottoms):
            if height < 4:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + height / 2,
                f"{height:.0f}%",
                ha="center",
                va="center",
                color="black" if segment_label != "NOVEL" else "white",
                fontweight="bold",
            )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_ylabel("Share of claims (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    ax.set_title("Redundancy Classifications")
    fig.tight_layout()
    writer.save("redundancy_stack")


def plot_efficiency_relevancy_scatter(summary: dict, writer: FigureWriter) -> None:
    plt = writer.plt
    our = summary["methods"][METHOD_OUR]
    baseline = summary["methods"][METHOD_BASELINE]

    points = [
        (
            METHOD_OUR,
            our["avg_efficiency_per_100_words"],
            pct(our["avg_relevancy_score"]),
        ),
        (
            METHOD_BASELINE,
            baseline["avg_efficiency_per_100_words"],
            pct(baseline["avg_relevancy_score"]),
        ),
    ]

    delta_relevancy = points[0][2] - points[1][2]
    delta_efficiency = points[0][1] - points[1][1]

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {METHOD_OUR: COLOR_OUR, METHOD_BASELINE: COLOR_BASELINE}

    for method, eff, rel in points:
        ax.scatter(
            eff,
            rel,
            s=220,
            color=colors[method],
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
            label=method_display(method),
        )
        ax.annotate(
            method_display(method),
            (eff, rel),
            textcoords="offset points",
            xytext=(12, 8 if method == METHOD_OUR else -16),
        )

    ax.annotate(
        "",
        xy=(points[0][1], points[0][2]),
        xytext=(points[1][1], points[1][2]),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        zorder=2,
    )

    ax.text(
        0.04,
        0.06,
        f"$\\Delta$ Relevancy = +{delta_relevancy:.2f} pp\n"
        f"$\\Delta$ Information Efficiency = +{delta_efficiency:.4f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("Information Efficiency (claims per 100 words)")
    ax.set_ylabel("Relevancy (%)")
    ax.set_title("Information Efficiency vs. Relevancy")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    writer.save("efficiency_relevancy")


def score_case_study(evaluation: dict) -> tuple[int, int]:
    """Prefer CAID visual-only noise with a fully relevant ViGALT output."""
    assets = evaluation["methods"][METHOD_BASELINE]["relevancy"]
    vig = evaluation["methods"][METHOD_OUR]["relevancy"]
    if vig.get("relevancy_score") != 100.0:
        return -1, 0

    visual_only = sum(1 for c in assets["claims"] if c.get("label") == "V")
    if visual_only == 0:
        return -1, 0

    folder = int(evaluation["folder_number"])
    score = visual_only * 10
    if 1 <= folder <= 15:
        score += 20
    if 2 <= visual_only <= 5:
        score += 10
    if visual_only == 3:
        score += 15
    if visual_only > 5:
        score -= (visual_only - 5) * 15

    assets_score = float(assets.get("relevancy_score") or 0.0)
    score += int(round(100.0 - assets_score))
    return score, visual_only


def product_in_range(folder: int, folder_range: tuple[int, int]) -> bool:
    low, high = folder_range
    return low <= folder <= high


def select_case_study_in_range(
    evaluations: list[dict],
    folder_range: tuple[int, int],
    override: int | None,
) -> dict:
    if override is not None:
        if not product_in_range(override, folder_range):
            raise ValueError(
                f"Product {override} is outside folder range {folder_range[0]}–{folder_range[1]}"
            )
        for evaluation in evaluations:
            if evaluation["folder_number"] == override:
                return evaluation
        raise ValueError(f"No evaluation for product folder {override}")

    candidates = [
        ev
        for ev in evaluations
        if product_in_range(int(ev["folder_number"]), folder_range)
        and score_case_study(ev)[0] >= 0
    ]
    if not candidates:
        raise ValueError(
            f"No qualifying case study in folders {folder_range[0]}–{folder_range[1]}"
        )

    def rank_key(ev: dict) -> tuple[int, int, int]:
        score, visual_only = score_case_study(ev)
        return score, visual_only, int(ev["folder_number"])

    return max(candidates, key=rank_key)


def select_case_study_pair(
    evaluations: list[dict],
    clothing_override: int | None,
    furniture_override: int | None,
) -> tuple[dict, dict]:
    clothing = select_case_study_in_range(
        evaluations, CLOTHING_FOLDER_RANGE, clothing_override
    )
    furniture = select_case_study_in_range(
        evaluations, FURNITURE_FOLDER_RANGE, furniture_override
    )
    return clothing, furniture


def escape_latex(text: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("#", r"\#"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("_", r"\_"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    )
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_claim_span(alt_text: str, claim: str) -> tuple[int, int] | None:
    if not claim.strip():
        return None

    candidates = [claim.strip()]
    if candidates[0].lower().startswith("the "):
        candidates.append(candidates[0][4:].strip())
    if candidates[0].endswith("."):
        candidates.append(candidates[0][:-1].strip())

    alt_lower = alt_text.lower()
    for candidate in candidates:
        idx = alt_lower.find(candidate.lower())
        if idx >= 0:
            return idx, idx + len(candidate)

    claim_norm = normalize_for_match(claim)
    alt_norm = normalize_for_match(alt_text)
    idx = alt_norm.find(claim_norm)
    if idx >= 0:
        return idx, idx + len(claim_norm)

    words = [w for w in re.split(r"\W+", claim_norm) if len(w) > 3]
    if len(words) >= 2:
        pattern = r".{0,40}".join(re.escape(w) for w in words[:4])
        match = re.search(pattern, alt_text, flags=re.IGNORECASE)
        if match:
            return match.start(), match.end()
    return None


def merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s[0], s[1]))
    merged: list[tuple[int, int, str]] = [spans[0]]
    for start, end, color in spans[1:]:
        last_start, last_end, last_color = merged[-1]
        if color == last_color and start <= last_end:
            merged[-1] = (last_start, max(last_end, end), last_color)
        else:
            merged.append((start, end, color))
    return merged


def highlight_alt_latex(
    alt_text: str,
    claims: list[dict],
    label: str,
    latex_color: str,
) -> str:
    spans: list[tuple[int, int, str]] = []
    for claim in claims:
        if claim.get("label") != label:
            continue
        span = find_claim_span(alt_text, claim.get("claim", ""))
        if span:
            spans.append((span[0], span[1], latex_color))

    return _apply_highlight_spans(alt_text, spans)


def highlight_non_relevant_latex(
    alt_text: str,
    claims: list[dict],
    latex_color: str,
) -> str:
    spans: list[tuple[int, int, str]] = []
    for claim in claims:
        if claim.get("label") == "R":
            continue
        span = find_claim_span(alt_text, claim.get("claim", ""))
        if span:
            spans.append((span[0], span[1], latex_color))
    return _apply_highlight_spans(alt_text, spans)


def _apply_highlight_spans(
    alt_text: str,
    spans: list[tuple[int, int, str]],
) -> str:
    if not spans:
        return escape_latex(alt_text)

    merged = merge_spans(spans)
    parts: list[str] = []
    cursor = 0
    for start, end, color in merged:
        if start > cursor:
            parts.append(escape_latex(alt_text[cursor:start]))
        parts.append(
            f"\\textcolor{{{color}}}{{{escape_latex(alt_text[start:end])}}}"
        )
        cursor = end
    if cursor < len(alt_text):
        parts.append(escape_latex(alt_text[cursor:]))
    return "".join(parts)


def count_non_relevant_claims(claims: list[dict]) -> int:
    return sum(1 for claim in claims if claim.get("label") != "R")


def latex_includegraphics_path(filename: str) -> str:
    """Filename for \\includegraphics (use with \\usepackage{grffile})."""
    return filename.replace("_", r"\_")


def resolve_product_image(
    dataset_dir: Path,
    folder: int,
    image_file: str,
) -> Path:
    candidates = (
        dataset_dir / str(folder) / image_file,
        dataset_dir / str(folder) / "image.jpg",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Product image not found for folder {folder} ({image_file}). "
        f"Checked: {', '.join(str(p) for p in candidates)}"
    )


def copy_case_study_images(
    evaluations: list[dict],
    dataset_dir: Path,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen_names: set[str] = set()

    for evaluation in evaluations:
        folder = int(evaluation["folder_number"])
        image_file = evaluation["image_file"]
        if image_file in seen_names:
            continue
        seen_names.add(image_file)

        source = resolve_product_image(dataset_dir, folder, image_file)
        destination = output_dir / image_file
        shutil.copy2(source, destination)
        copied.append(destination)

    return copied


def render_case_study_subfigure(
    evaluation: dict,
    subcaption: str,
    label_slug: str,
) -> str:
    image_file = evaluation["image_file"]
    our_alt = evaluation["methods"][METHOD_OUR]["alt_text"]
    base_alt = evaluation["methods"][METHOD_BASELINE]["alt_text"]
    base_claims = evaluation["methods"][METHOD_BASELINE]["relevancy"]["claims"]

    assets_highlighted = highlight_non_relevant_latex(base_alt, base_claims, "red")
    our_plain = escape_latex(our_alt)

    return f"""  \\begin{{subfigure}}[t]{{\\linewidth}}
    \\centering
    \\includegraphics[width=0.34\\linewidth]{{{latex_includegraphics_path(image_file)}}}
    \\\\[0.3em]
    {{\\scriptsize
    \\begin{{minipage}}[t]{{0.48\\linewidth}}
    \\textbf{{CAID:}}\\par
    \\vspace{{0.2em}}
    {assets_highlighted}
    \\end{{minipage}}\\hfill
    \\begin{{minipage}}[t]{{0.48\\linewidth}}
    \\textbf{{Our Pipeline:}}\\par
    \\vspace{{0.2em}}
    {our_plain}
    \\end{{minipage}}
    }}
    \\caption{{{escape_latex(subcaption)}}}
    \\label{{fig:case-study-{label_slug}}}
  \\end{{subfigure}}"""


def generate_case_study_latex(
    clothing_eval: dict,
    furniture_eval: dict,
    output_path: Path,
) -> None:
    cloth_id = clothing_eval["folder_number"]
    furn_id = furniture_eval["folder_number"]
    clothing_block = render_case_study_subfigure(
        clothing_eval, "Clothing", f"clothing-{cloth_id}"
    )
    furniture_block = render_case_study_subfigure(
        furniture_eval, "Furniture", f"furniture-{furn_id}"
    )

    cloth_img = clothing_eval["image_file"]
    furn_img = furniture_eval["image_file"]
    content = f"""% Requires: \\usepackage{{graphicx}}, \\usepackage{{grffile}}, \\usepackage{{xcolor}}, \\usepackage{{subcaption}}
% Product images (same directory as this file): {cloth_img}, {furn_img}
\\begin{{figure}}[t]
  \\centering
{clothing_block}

  \\vspace{{0.7em}}
{furniture_block}
  \\caption{{Qualitative comparison on clothing and furniture listings. \\textcolor{{red}}{{Red}} highlights purchase-irrelevant detail in CAID; our pipeline text is unmarked.}}
  \\label{{fig:case-study-clothing-{cloth_id}-furniture-{furn_id}}}
\\end{{figure}}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def should_run(only: str, name: str) -> bool:
    return only == "all" or only == name


def main() -> int:
    configure_stdout()
    args = parse_args()

    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir)
    summary = load_summary(dataset_dir)
    evaluations = discover_evaluations(dataset_dir)
    breakdown = micro_redundancy_breakdown(evaluations)

    run_plots = (
        should_run(args.only, "core_metrics")
        or should_run(args.only, "redundancy_stack")
        or should_run(args.only, "efficiency_relevancy")
    )
    if run_plots:
        plt_module, _ = require_matplotlib()
        configure_plotting()
        writer = FigureWriter(output_dir, args.formats, plt_module)

        if should_run(args.only, "core_metrics"):
            plot_core_metrics(summary, writer)

        if should_run(args.only, "redundancy_stack"):
            plot_redundancy_stack(breakdown, writer)

        if should_run(args.only, "efficiency_relevancy"):
            plot_efficiency_relevancy_scatter(summary, writer)

    if should_run(args.only, "case_study") and not args.skip_case_study:
        clothing_eval, furniture_eval = select_case_study_pair(
            evaluations,
            args.case_study_clothing,
            args.case_study_furniture,
        )
        for label, case_eval in (
            ("Clothing", clothing_eval),
            ("Furniture", furniture_eval),
        ):
            folder = case_eval["folder_number"]
            score, _ = score_case_study(case_eval)
            base_claims = case_eval["methods"][METHOD_BASELINE]["relevancy"]["claims"]
            non_relevant = count_non_relevant_claims(base_claims)
            print(
                f"Case study {label}: product {folder} "
                f"(score={score}, CAID non-relevant highlighted spans={non_relevant})",
                file=sys.stderr,
            )

        tex_path = output_dir / "case_study_figure.tex"
        generate_case_study_latex(
            clothing_eval,
            furniture_eval,
            tex_path,
        )
        copied_images = copy_case_study_images(
            [clothing_eval, furniture_eval],
            dataset_dir,
            output_dir,
        )
        print(f"Wrote {tex_path}", file=sys.stderr)
        for image_path in copied_images:
            print(f"Copied {image_path.name} -> {image_path}", file=sys.stderr)

    print(f"Figures written to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
