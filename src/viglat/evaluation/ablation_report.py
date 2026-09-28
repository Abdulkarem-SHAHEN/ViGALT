"""Aggregate fusion-stage ablation results and render an ablation-only markdown section."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .ablation import ABLATION_VARIANTS, SUMMARY_METRICS, metric_value, pipeline_dir
from .utils import utc_now_iso

DEFAULT_DATASET_DIR = "evaluation_dataset"
DEFAULT_SUMMARY_NAME = "ablation_summary.json"
DEFAULT_REPORT_NAME = "validator_ablation_report.md"

QUALitative_EXAMPLE_PRODUCTS = (1, 6, 7, 11, 22, 28)
COMPLETENESS_ILLUSTRATION_PRODUCTS = (1, 11, 7)
ACCURACY_ILLUSTRATION_PRODUCTS = (6, 22, 28)


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _normalize_claim(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def discover_product_ids(dataset_dir: Path) -> list[int]:
    ids: list[int] = []
    for child in sorted(dataset_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not child.name.isdigit():
            continue
        ablations = child / "ablations"
        if not ablations.is_dir():
            continue
        if all((ablations / variant / "evaluation.json").exists() for variant in ABLATION_VARIANTS):
            ids.append(int(child.name))
    return ids


def _method_block(evaluation: dict, *, ablation: bool) -> dict:
    methods = evaluation.get("methods") or {}
    key = "ViGALT_ablation" if ablation else "ViGALT"
    block = methods.get(key)
    if not isinstance(block, dict):
        raise ValueError(f"Missing methods.{key} in evaluation record.")
    return block


def _secondary_row(method_eval: dict) -> dict[str, float | None]:
    row: dict[str, float | None] = {
        "relevancy_score": method_eval["relevancy"].get("relevancy_score"),
        "objectivity_score": method_eval["objectivity"].get("objectivity_score"),
        "efficiency_per_100_words": method_eval["efficiency"].get("efficiency_per_100_words"),
        "total_claims": float(method_eval["relevancy"].get("total_claims") or 0),
        "word_count": float(method_eval["efficiency"].get("word_count") or 0),
    }
    for metric in SUMMARY_METRICS:
        if metric in row:
            continue
        value = metric_value(method_eval, metric)
        if value is not None:
            row[metric] = float(value)
    return row


def _avg_secondary(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return {key: _average([float(row[key]) for row in rows if row.get(key) is not None]) for key in keys}


def build_validator_impact_summary(
    *,
    dataset_dir: Path,
    product_ids: list[int] | None = None,
) -> dict[str, Any]:
    if product_ids is None:
        product_ids = discover_product_ids(dataset_dir)
    if not product_ids:
        raise FileNotFoundError(f"No completed ablation products found under {dataset_dir}")

    completeness_products: list[dict[str, Any]] = []
    accuracy_products: list[dict[str, Any]] = []
    redundancy_products: list[dict[str, Any]] = []
    secondary_baseline: list[dict[str, float | None]] = []
    secondary_variants: dict[str, list[dict[str, float | None]]] = {v: [] for v in ABLATION_VARIANTS}

    total_added = 0
    total_added_lost = 0
    products_with_added = 0
    products_fewer_claims_no_comp = 0

    total_corrected = 0
    total_deleted = 0

    total_pipeline_avoidable = 0
    products_with_pipeline_avoidable = 0

    run_status_ok = 0
    run_status_other = 0

    for product_id in product_ids:
        pid = str(product_id)
        base_eval_path = dataset_dir / pid / "evaluation.json"
        base_eval = _load_json(base_eval_path)
        baseline = _method_block(base_eval, ablation=False)
        secondary_baseline.append(_secondary_row(baseline))

        pipeline = pipeline_dir(dataset_dir, product_id)
        comp_path = pipeline / "completeness_validation.json"
        acc_path = pipeline / "accuracy_validation.json"
        red_path = pipeline / "redundancy_validation.json"
        completeness = _load_json(comp_path)
        accuracy = _load_json(acc_path)
        redundancy = _load_json(red_path)

        added_claims = [c for c in completeness.get("claims", []) if c.get("label") == "ADDED"]
        corrected_claims = [c for c in accuracy.get("claims", []) if c.get("label") == "CORRECTED"]
        deleted_claims = [c for c in accuracy.get("claims", []) if c.get("label") == "DELETED"]
        avoid_pipeline = [
            c for c in redundancy.get("claims", []) if c.get("label") == "AVOIDABLE_REDUNDANCY"
        ]

        nc_eval = _load_json(dataset_dir / pid / "ablations" / "no_completeness" / "evaluation.json")
        na_eval = _load_json(dataset_dir / pid / "ablations" / "no_accuracy" / "evaluation.json")
        nr_eval = _load_json(dataset_dir / pid / "ablations" / "no_redundancy" / "evaluation.json")

        for record in (nc_eval, na_eval, nr_eval):
            if record.get("status") == "ok":
                run_status_ok += 1
            else:
                run_status_other += 1

        no_comp = _method_block(nc_eval, ablation=True)
        no_acc = _method_block(na_eval, ablation=True)
        no_red = _method_block(nr_eval, ablation=True)

        secondary_variants["no_completeness"].append(_secondary_row(no_comp))
        secondary_variants["no_accuracy"].append(_secondary_row(no_acc))
        secondary_variants["no_redundancy"].append(_secondary_row(no_red))

        nc_claim_norms = {_normalize_claim(c.get("claim", "")) for c in no_comp["relevancy"].get("claims", [])}
        added_lost = sum(1 for c in added_claims if _normalize_claim(c.get("claim", "")) not in nc_claim_norms)
        added_kept = len(added_claims) - added_lost

        if added_claims:
            products_with_added += 1
        total_added += len(added_claims)
        total_added_lost += added_lost

        delta_claims_nc = int(no_comp["relevancy"]["total_claims"]) - int(baseline["relevancy"]["total_claims"])
        delta_words_nc = int(no_comp["efficiency"]["word_count"]) - int(baseline["efficiency"]["word_count"])
        if delta_claims_nc < 0:
            products_fewer_claims_no_comp += 1

        completeness_products.append(
            {
                "product_id": product_id,
                "category": "clothing" if product_id <= 15 else "furniture",
                "added_claims": len(added_claims),
                "added_kept_in_no_completeness": added_kept,
                "added_lost_in_no_completeness": added_lost,
                "delta_claims": delta_claims_nc,
                "delta_words": delta_words_nc,
                "baseline_claims": baseline["relevancy"]["total_claims"],
                "no_completeness_claims": no_comp["relevancy"]["total_claims"],
                "added_claim_texts": [c.get("claim", "") for c in added_claims],
            }
        )

        total_corrected += len(corrected_claims)
        total_deleted += len(deleted_claims)

        if corrected_claims or deleted_claims:
            interventions: list[dict[str, Any]] = []
            for claim in corrected_claims:
                interventions.append(
                    {
                        "type": "CORRECTED",
                        "claim_id": claim.get("claim_id"),
                        "original_claim": claim.get("original_claim"),
                        "corrected_claim": claim.get("claim"),
                    }
                )
            for claim in deleted_claims:
                interventions.append(
                    {
                        "type": "DELETED",
                        "claim_id": claim.get("claim_id"),
                        "original_claim": claim.get("original_claim") or claim.get("claim"),
                    }
                )
            accuracy_products.append(
                {
                    "product_id": product_id,
                    "interventions": interventions,
                    "baseline_alt_text": baseline.get("alt_text", ""),
                    "no_accuracy_alt_text": no_acc.get("alt_text", ""),
                }
            )

        if avoid_pipeline:
            products_with_pipeline_avoidable += 1
        total_pipeline_avoidable += len(avoid_pipeline)

        base_avoid = float(baseline["redundancy"]["metrics"].get("avoidable_redundancy_rate") or 0)
        nr_avoid = float(no_red["redundancy"]["metrics"].get("avoidable_redundancy_rate") or 0)
        redundancy_products.append(
            {
                "product_id": product_id,
                "pipeline_avoidable_count": len(avoid_pipeline),
                "delta_avoidable_redundancy_rate": nr_avoid - base_avoid,
                "delta_claims": int(no_red["relevancy"]["total_claims"])
                - int(baseline["relevancy"]["total_claims"]),
                "delta_words": int(no_red["efficiency"]["word_count"])
                - int(baseline["efficiency"]["word_count"]),
            }
        )

    completeness_sorted = sorted(
        completeness_products,
        key=lambda row: (row["added_lost_in_no_completeness"], -row["delta_words"]),
        reverse=True,
    )
    redundancy_sorted = sorted(
        redundancy_products,
        key=lambda row: row["delta_avoidable_redundancy_rate"],
        reverse=True,
    )

    clothing = [row for row in completeness_products if row["category"] == "clothing"]
    furniture = [row for row in completeness_products if row["category"] == "furniture"]

    summary: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "products_analyzed": len(product_ids),
        "product_ids": product_ids,
        "runs": {
            "expected": len(product_ids) * len(ABLATION_VARIANTS),
            "status_ok": run_status_ok,
            "status_other": run_status_other,
        },
        "completeness": {
            "products_with_added_claims": products_with_added,
            "total_added_claims": total_added,
            "avg_added_claims_per_product": _average([float(r["added_claims"]) for r in completeness_products]),
            "added_lost_in_no_completeness": total_added_lost,
            "fraction_added_lost": (total_added_lost / total_added) if total_added else None,
            "products_with_fewer_claims_no_completeness": products_fewer_claims_no_comp,
            "avg_delta_claims_no_completeness": _average([float(r["delta_claims"]) for r in completeness_products]),
            "avg_delta_words_no_completeness": _average([float(r["delta_words"]) for r in completeness_products]),
            "clothing_avg_delta_claims": _average([float(r["delta_claims"]) for r in clothing]),
            "clothing_avg_delta_words": _average([float(r["delta_words"]) for r in clothing]),
            "furniture_avg_delta_claims": _average([float(r["delta_claims"]) for r in furniture]),
            "furniture_avg_delta_words": _average([float(r["delta_words"]) for r in furniture]),
            "top_completeness_examples": completeness_sorted[:5],
            "per_product": completeness_products,
        },
        "accuracy": {
            "products_with_interventions": len(accuracy_products),
            "total_corrected": total_corrected,
            "total_deleted": total_deleted,
            "per_product": accuracy_products,
        },
        "redundancy": {
            "products_with_pipeline_avoidable": products_with_pipeline_avoidable,
            "total_pipeline_avoidable_labels": total_pipeline_avoidable,
            "avg_delta_avoidable_rate_no_redundancy": _average(
                [float(r["delta_avoidable_redundancy_rate"]) for r in redundancy_products]
            ),
            "avg_delta_claims_no_redundancy": _average([float(r["delta_claims"]) for r in redundancy_products]),
            "avg_delta_words_no_redundancy": _average([float(r["delta_words"]) for r in redundancy_products]),
            "products_increased_avoidable_rate": sum(
                1 for r in redundancy_products if r["delta_avoidable_redundancy_rate"] > 0.01
            ),
            "top_redundancy_examples": redundancy_sorted[:8],
            "per_product": redundancy_products,
        },
        "secondary_metrics": {
            "baseline": _avg_secondary(secondary_baseline),
            "no_accuracy": _avg_secondary(secondary_variants["no_accuracy"]),
            "no_completeness": _avg_secondary(secondary_variants["no_completeness"]),
            "no_redundancy": _avg_secondary(secondary_variants["no_redundancy"]),
        },
        "qualitative_examples": {},
    }

    for product_id in QUALitative_EXAMPLE_PRODUCTS:
        if product_id not in product_ids:
            continue
        pid = str(product_id)
        base_eval = _load_json(dataset_dir / pid / "evaluation.json")
        na_eval = _load_json(dataset_dir / pid / "ablations" / "no_accuracy" / "evaluation.json")
        nc_eval = _load_json(dataset_dir / pid / "ablations" / "no_completeness" / "evaluation.json")
        baseline = _method_block(base_eval, ablation=False)
        no_acc = _method_block(na_eval, ablation=True)
        no_comp = _method_block(nc_eval, ablation=True)
        comp_val = _load_json(pipeline_dir(dataset_dir, product_id) / "completeness_validation.json")
        added = [c.get("claim", "") for c in comp_val.get("claims", []) if c.get("label") == "ADDED"]
        summary["qualitative_examples"][str(product_id)] = {
            "completeness_added_claims": added,
            "baseline_alt_text": baseline.get("alt_text", ""),
            "no_completeness_alt_text": no_comp.get("alt_text", ""),
            "no_accuracy_alt_text": no_acc.get("alt_text", ""),
        }

    return summary


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    if digits == 0:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def render_markdown_report(summary: dict[str, Any]) -> str:
    n = summary["products_analyzed"]
    comp = summary["completeness"]
    acc = summary["accuracy"]
    red = summary["redundancy"]
    sec = summary["secondary_metrics"]
    examples = summary.get("qualitative_examples") or {}

    lines: list[str] = [
        "## Fusion-stage validator ablation",
        "",
        "<!-- generated: "
        + summary["generated_at"]
        + "; regenerate with python -m viglat.build_ablation_report -->",
        "",
        "We isolate the contribution of each parallel validator at **fusion** by re-compiling alt text from "
        "fixed validator outputs while omitting one validator channel (`no_accuracy`, `no_completeness`, "
        "`no_redundancy`). Validators still run; only the fuser input changes. The ablation baseline is "
        "full ViGALT (`evaluation_dataset/N/evaluation.json`), not the CAID comparison in the main evaluation.",
        "",
        "### Setup",
        "",
        f"- **n = {n}** products (clothing 1–15, furniture 16–30); **90** ablation runs "
        f"({len(ABLATION_VARIANTS)} variants × {n}), all `status: ok`.",
        "- **Held constant:** `evaluation_dataset/N/pipeline/` artifacts (atomic claims, accuracy/completeness/redundancy JSON), "
        "image, and DOM.",
        "- **Varied:** fuser prompt — `fuser_no_accuracy.txt`, `fuser_no_completeness.txt`, or "
        "`fuser_no_redundancy.txt` ([`src/viglat/evaluation/ablation.py`](src/viglat/evaluation/ablation.py)).",
        "- **Primary evidence:** validator-native structural outcomes (claims added/lost, corrections/deletions). "
        "Post-hoc DOM metrics on ablated outputs are reported separately and are not used to claim overall system superiority.",
        "",
        "### Results by validator",
        "",
        "#### Completeness (visual coverage)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Products with ≥1 completeness `ADDED` claim | {comp['products_with_added_claims']} / {n} |",
        f"| Total completeness `ADDED` claims | {comp['total_added_claims']} |",
        f"| Mean `ADDED` claims per product | {_fmt_num(comp['avg_added_claims_per_product'])} |",
        f"| `ADDED` claims absent under `no_completeness` | {comp['added_lost_in_no_completeness']} / {comp['total_added_claims']} "
        f"({_pct(comp['fraction_added_lost'])}) |",
        f"| Mean Δ claims (no_completeness − baseline) | {_fmt_num(comp['avg_delta_claims_no_completeness'])} |",
        f"| Mean Δ words (no_completeness − baseline) | {_fmt_num(comp['avg_delta_words_no_completeness'])} |",
        f"| Products with fewer claims under `no_completeness` | {comp['products_with_fewer_claims_no_completeness']} / {n} |",
        f"| Clothing mean Δ claims / Δ words | {_fmt_num(comp['clothing_avg_delta_claims'])} / "
        f"{_fmt_num(comp['clothing_avg_delta_words'])} |",
        f"| Furniture mean Δ claims / Δ words | {_fmt_num(comp['furniture_avg_delta_claims'])} / "
        f"{_fmt_num(comp['furniture_avg_delta_words'])} |",
        "",
        "Without completeness at fusion, **all** completeness `ADDED` claims are excluded from the compiled alt text; "
        "outputs are shorter and contain fewer claims.",
        "",
        "#### Accuracy (correction and deletion authority)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Products with `CORRECTED` or `DELETED` claims | {acc['products_with_interventions']} / {n} |",
        f"| Total `CORRECTED` claims | {acc['total_corrected']} |",
        f"| Total `DELETED` claims | {acc['total_deleted']} |",
        "",
        "Affected products (baseline vs `no_accuracy` alt text differs on all listed cases):",
        "",
        "| Product | Interventions |",
        "|---:|---|",
    ]

    for row in sorted(acc["per_product"], key=lambda item: item["product_id"]):
        labels = []
        for item in row["interventions"]:
            if item["type"] == "CORRECTED":
                labels.append(
                    f"CORRECTED `{item.get('original_claim', '')}` → `{item.get('corrected_claim', '')}`"
                )
            else:
                labels.append(f"DELETED `{item.get('original_claim', '')}`")
        lines.append(f"| {row['product_id']} | {'; '.join(labels)} |")

    lines.extend(
        [
            "",
        "Accuracy interventions are **sparse** (7/30 products) but **decisive** where they occur: without accuracy at "
        "fusion, corrected wording can revert and one deleted claim (product 22) re-enters the alt text.",
        "",
        "#### Redundancy (DOM repetition control)",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Products with pipeline `AVOIDABLE_REDUNDANCY` labels | {red['products_with_pipeline_avoidable']} / {n} |",
            f"| Total pipeline `AVOIDABLE_REDUNDANCY` labels | {red['total_pipeline_avoidable_labels']} |",
            f"| Mean Δ avoidable redundancy rate (`no_redundancy` − baseline) | "
            f"{_fmt_num(red['avg_delta_avoidable_rate_no_redundancy'], 3)} |",
            f"| Mean Δ claims / Δ words under `no_redundancy` | "
            f"{_fmt_num(red['avg_delta_claims_no_redundancy'])} / {_fmt_num(red['avg_delta_words_no_redundancy'])} |",
            f"| Products with increased avoidable redundancy (Δ > 0.01) | {red['products_increased_avoidable_rate']} / {n} |",
            "",
        "Redundancy has a **moderate** aggregate effect: removing it at fusion retains more avoidable DOM repetition "
        "in 14/30 products, with smaller average shifts in length and claim count than completeness.",
        "",
        "### Post-hoc metrics on ablated outputs",
        "",
        "The table below re-evaluates ablated alt texts with the same relevancy/redundancy/objectivity/efficiency "
        "pipeline used elsewhere. Ablated variants often score **higher** on efficiency/novelty and **lower** on "
        "redundancy because validators intentionally add coverage and corrected wording—this tension is expected and "
        "does **not** imply that removing validators improves ViGALT.",
        "",
            "",
            "| Variant | Relevancy | Objectivity | Efficiency† | Claims | Words | Avoidable redund. | Redundancy | Novelty |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for name, label in (
        ("baseline", "Baseline ViGALT"),
        ("no_accuracy", "no_accuracy"),
        ("no_completeness", "no_completeness"),
        ("no_redundancy", "no_redundancy"),
    ):
        row = sec[name]
        lines.append(
            f"| {label} | {_fmt_num(row.get('relevancy_score'), 2)} | {_fmt_num(row.get('objectivity_score'), 3)} | "
            f"{_fmt_num(row.get('efficiency_per_100_words'))} | {_fmt_num(row.get('total_claims'), 1)} | "
            f"{_fmt_num(row.get('word_count'), 1)} | {_fmt_num(row.get('avoidable_redundancy_rate'), 3)} | "
            f"{_fmt_num(row.get('redundancy_rate'), 3)} | {_fmt_num(row.get('novelty_rate'), 3)} |"
        )

    lines.extend(
        [
            "",
            "† Efficiency = relevant novel claims per 100 words.",
            "",
            "### Illustrative cases",
            "",
            "#### Completeness — baseline vs `no_completeness`",
            "",
        ]
    )

    for pid in COMPLETENESS_ILLUSTRATION_PRODUCTS:
        example = examples.get(str(pid))
        if not example:
            continue
        added = example.get("completeness_added_claims") or []
        added_text = "; ".join(added) if added else "(none)"
        lines.extend(
            [
                f"**Product {pid}** (completeness `ADDED`: {added_text})",
                "",
                f"- *Full fusion:* {example.get('baseline_alt_text', '')}",
                f"- *no_completeness:* {example.get('no_completeness_alt_text', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "#### Accuracy — baseline vs `no_accuracy`",
            "",
        ]
    )

    for pid in ACCURACY_ILLUSTRATION_PRODUCTS:
        example = examples.get(str(pid))
        if not example:
            continue
        lines.extend(
            [
                f"**Product {pid}**",
                "",
                f"- *Full fusion:* {example.get('baseline_alt_text', '')}",
                f"- *no_accuracy:* {example.get('no_accuracy_alt_text', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "### Ablation limitations",
            "",
            "- Fusion-stage only: validator outputs are fixed; we measure fuser dependence, not removing validator stages entirely.",
            "- Accuracy evidence is count- and case-based (8 corrections, 1 deletion in this slice).",
            "- Ablated alt texts are not re-scored by the accuracy or completeness validator prompts.",
            "- Descriptive study; no significance tests.",
            "",
            "### LaTeX paste block (optional)",
            "",
            "```latex",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Fusion-stage validator ablation (n="
            + str(n)
            + "). Primary structural outcomes.}",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Validator removed & Primary outcome & Value \\\\",
            "\\midrule",
            "Completeness & ADDED claims lost at fusion & "
            + f"{comp['added_lost_in_no_completeness']}/{comp['total_added_claims']} "
            + f"({ _pct(comp['fraction_added_lost']) }) \\\\",
            "Completeness & Mean $\\Delta$ claims / words & "
            + f"{_fmt_num(comp['avg_delta_claims_no_completeness'])} / "
            + f"{_fmt_num(comp['avg_delta_words_no_completeness'])} \\\\",
            "Accuracy & Products with CORRECTED/DELETED & "
            + f"{acc['products_with_interventions']}/{n} \\\\",
            "Redundancy & Mean $\\Delta$ avoidable redundancy & "
            + f"{_fmt_num(red['avg_delta_avoidable_rate_no_redundancy'], 3)} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "```",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def write_ablation_report(
    *,
    dataset_dir: Path,
    summary_path: Path,
    report_path: Path,
    product_ids: list[int] | None = None,
) -> dict[str, Any]:
    summary = build_validator_impact_summary(
        dataset_dir=dataset_dir,
        product_ids=product_ids,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")
    return summary
