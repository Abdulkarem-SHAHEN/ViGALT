"""Aggregate evaluation.json records into summary.json-style metrics."""

from __future__ import annotations

MAIN_DATASET_METHODS = ("ViGALT", "CAID")

SUMMARY_METRICS = (
    "relevancy_score",
    "objectivity_score",
    "efficiency_per_100_words",
    "novelty_rate",
    "raw_novelty_rate",
    "avoidable_redundancy_rate",
    "unavoidable_repetition_rate",
    "redundancy_rate",
)


def metric_value(method_eval: dict, metric: str) -> float | None:
    if metric == "relevancy_score":
        score = method_eval["relevancy"].get("relevancy_score")
        if score is None:
            return None
        return float(score) / 100.0
    if metric == "objectivity_score":
        return method_eval["objectivity"].get("objectivity_score")
    if metric == "efficiency_per_100_words":
        return method_eval["efficiency"].get("efficiency_per_100_words")
    if metric in {
        "novelty_rate",
        "raw_novelty_rate",
        "avoidable_redundancy_rate",
        "unavoidable_repetition_rate",
        "redundancy_rate",
    }:
        return method_eval["redundancy"]["metrics"].get(metric)
    raise ValueError(f"Unknown summary metric: {metric}")


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_main_dataset_summary(
    all_evaluations: list[dict],
    *,
    methods: tuple[str, ...] = MAIN_DATASET_METHODS,
) -> dict:
    summary: dict = {"folders_counted": len(all_evaluations), "methods": {}}

    for method in methods:
        method_values: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
        total_claims = 0

        for evaluation in all_evaluations:
            method_eval = evaluation["methods"][method]
            total_claims += method_eval["relevancy"]["total_claims"]
            for metric in SUMMARY_METRICS:
                value = metric_value(method_eval, metric)
                if value is not None:
                    method_values[metric].append(float(value))

        summary["methods"][method] = {
            "total_claims": total_claims,
            **{
                f"avg_{metric}": average(method_values[metric])
                for metric in SUMMARY_METRICS
            },
        }

    return summary
