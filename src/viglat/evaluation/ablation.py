"""Validator ablation study: fuse without one validator, then evaluate."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI

from ..backend.config import DEFAULT_EVALUATION_MODEL, DEFAULT_OPENAI_MODEL
from ..backend.openai import run_stage_openai
from ..paths import project_root
from ..pipeline.types import StageConfig
from ..run_layout import IMAGE_BASENAME
from .alt_to_list import CLAIM_EXTRACTION_STAGE
from .objectivity import OBJECTIVITY_STAGE, ObjectivityJob, build_result_record as build_objectivity_record
from .redundancy import REDUNDANCY_STAGE, RedundancyJob, build_result_record as build_redundancy_record
from .relevancy import RELEVANCY_STAGE, RelevancyJob, build_result_record as build_relevancy_record
from .dataset_summary import SUMMARY_METRICS, average, metric_value
from .utils import (
    IMAGE_EXTENSIONS,
    build_relevancy_block,
    calculate_efficiency,
    count_alt_words,
    image_to_data_url,
    load_product_dom,
    save_output,
    utc_now_iso,
)

AblationVariant = Literal["no_accuracy", "no_completeness", "no_redundancy"]

PROMPTS_DIR = "prompts"
DEFAULT_VALIDATIONS_DIR = "ViGALT_validations"
DEFAULT_DATASET_DIR = "evaluation_dataset"

ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    "no_accuracy",
    "no_completeness",
    "no_redundancy",
)

ABLATION_CONFIG: dict[AblationVariant, dict] = {
    "no_accuracy": {
        "prompt_file": "fuser_no_accuracy.txt",
        "token_keys": ("ALT_TEXT", "ATOMIC_CLAIMS", "COMPLETENESS_OUTPUT", "REDUNDANCY_OUTPUT"),
    },
    "no_completeness": {
        "prompt_file": "fuser_no_completeness.txt",
        "token_keys": ("ALT_TEXT", "ATOMIC_CLAIMS", "ACCURACY_OUTPUT", "REDUNDANCY_OUTPUT"),
    },
    "no_redundancy": {
        "prompt_file": "fuser_no_redundancy.txt",
        "token_keys": ("ALT_TEXT", "ATOMIC_CLAIMS", "ACCURACY_OUTPUT", "COMPLETENESS_OUTPUT"),
    },
}


@dataclass(frozen=True)
class AblationJob:
    product_id: int
    variant: AblationVariant
    validations_dir: Path
    dataset_dir: Path
    output_dir: Path


def parse_product_ids(value: str | None, *, max_product: int = 30) -> list[int]:
    if not value:
        return list(range(1, max_product + 1))

    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            ids.extend(range(int(start_s), int(end_s) + 1))
        else:
            ids.append(int(part))
    return sorted({product_id for product_id in ids if 1 <= product_id <= max_product})


def parse_variants(value: str | None) -> list[AblationVariant]:
    if not value:
        return list(ABLATION_VARIANTS)
    variants: list[AblationVariant] = []
    for part in value.split(","):
        name = part.strip()
        if name not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant: {name!r}")
        variants.append(name)  # type: ignore[arg-type]
    return variants


def category_for_product(product_id: int) -> str:
    return "CLOTHING" if product_id <= 15 else "FURNITURE"


def find_validation_image(folder: Path) -> Path:
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No product image found in {folder}")


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def load_validation_inputs(validations_dir: Path, product_id: int) -> dict:
    folder = validations_dir / str(product_id)
    if not folder.is_dir():
        raise FileNotFoundError(f"Validation folder not found: {folder}")

    alt_text_path = folder / "generated_alt_text.txt"
    if not alt_text_path.exists():
        raise FileNotFoundError(f"Missing generated_alt_text.txt in {folder}")

    claims = load_json(folder / "claims.json")
    accuracy = load_json(folder / "accuracy_validation.json")
    completeness = load_json(folder / "completeness_validation.json")
    redundancy = load_json(folder / "redundancy_validation.json")
    image_path = find_validation_image(folder)

    atomic_claims = claims.get("claims")
    if not isinstance(atomic_claims, list) or not atomic_claims:
        raise ValueError(f"Missing claims in {folder / 'claims.json'}")

    return {
        "alt_text": alt_text_path.read_text(encoding="utf-8").strip(),
        "atomic_claims": atomic_claims,
        "accuracy_output": accuracy,
        "completeness_output": completeness,
        "redundancy_output": redundancy,
        "image_path": image_path,
    }


def build_fusion_tokens(variant: AblationVariant, inputs: dict) -> dict:
    source_map = {
        "ALT_TEXT": inputs["alt_text"],
        "ATOMIC_CLAIMS": inputs["atomic_claims"],
        "ACCURACY_OUTPUT": inputs["accuracy_output"],
        "COMPLETENESS_OUTPUT": inputs["completeness_output"],
        "REDUNDANCY_OUTPUT": inputs["redundancy_output"],
    }
    keys = ABLATION_CONFIG[variant]["token_keys"]
    return {key: source_map[key] for key in keys}


def fusion_stage_for_variant(variant: AblationVariant) -> StageConfig:
    return StageConfig(
        name="fusion",
        prompt_file=ABLATION_CONFIG[variant]["prompt_file"],
        model=DEFAULT_OPENAI_MODEL,
        uses_image=False,
        output_type="json",
        expected_stage="fusion",
    )


def run_fusion(
    *,
    client: OpenAI,
    variant: AblationVariant,
    inputs: dict,
    prompts_dir: Path,
    fuser_model: str,
    timeout_s: float,
) -> dict:
    stage = fusion_stage_for_variant(variant)
    result = run_stage_openai(
        client,
        stage=stage,
        tokens=build_fusion_tokens(variant, inputs),
        prompts_dir=prompts_dir,
        image_url="",
        single_model=fuser_model,
        timeout_s=timeout_s,
    )
    if result.status != "ok" or not isinstance(result.parsed, dict):
        raise RuntimeError(result.error or f"Fusion failed for variant {variant}")

    final_alt_text = result.parsed.get("final_alt_text")
    if not isinstance(final_alt_text, str) or not final_alt_text.strip():
        raise ValueError("Fusion output is missing final_alt_text.")

    return {
        "status": "ok",
        "ablation": variant,
        "model": fuser_model,
        "prompt_file": stage.prompt_file,
        "duration_ms": result.duration_ms,
        "token_usage": result.token_usage,
        "raw_response": result.raw_response,
        "parsed": result.parsed,
        "final_alt_text": final_alt_text.strip(),
        "processed_at": utc_now_iso(),
    }


def run_claim_extraction(
    *,
    client: OpenAI,
    final_alt_text: str,
    product_id: int,
    category: str,
    prompts_dir: Path,
    claim_model: str,
    timeout_s: float,
) -> dict:
    result = run_stage_openai(
        client,
        stage=CLAIM_EXTRACTION_STAGE,
        tokens={"ALT_TEXT": final_alt_text},
        prompts_dir=prompts_dir,
        image_url="",
        single_model=claim_model,
        timeout_s=timeout_s,
    )
    if result.status != "ok" or not isinstance(result.parsed, dict):
        raise RuntimeError(result.error or "Claim extraction failed")

    claims = result.parsed.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Claim extraction returned no claims.")

    product_stem = str(product_id)
    return {
        "status": "ok",
        "source_kind": "ablation_fusion",
        "stem": product_stem,
        "product_stem": product_stem,
        "alt_text": final_alt_text,
        "final_alt_text": final_alt_text,
        "classification_category": category,
        "model": claim_model,
        "prompt_file": CLAIM_EXTRACTION_STAGE.prompt_file,
        "duration_ms": result.duration_ms,
        "token_usage": result.token_usage,
        "raw_response": result.raw_response,
        "parsed": result.parsed,
        "processed_at": utc_now_iso(),
    }


def ensure_eval_image(source_image: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_image.suffix.lower() or ".jpg"
    target = output_dir / f"{IMAGE_BASENAME}{suffix}"
    if not target.exists() or target.stat().st_size == 0:
        shutil.copy2(source_image, target)
    return target.resolve()


def run_evaluations(
    *,
    client: OpenAI,
    job: AblationJob,
    claims_record: dict,
    final_alt_text: str,
    eval_image_dir: Path,
    products_dir: Path,
    prompts_dir: Path,
    eval_model: str,
    timeout_s: float,
) -> dict:
    product_stem = str(job.product_id)
    category = category_for_product(job.product_id)
    claims = claims_record["parsed"]["claims"]
    product_dom = load_product_dom(product_stem, products_dir)

    relevancy_job = RelevancyJob(
        algorithm="ViGALT",
        claim_file=job.output_dir / "claims.json",
        product_stem=product_stem,
        category=category,
        claims=claims,
        metadata={"vigalt_alt_text": final_alt_text, "final_alt_text": final_alt_text},
    )
    image_url, image_reference = _resolve_image(eval_image_dir, products_dir, product_stem)
    relevancy_result = run_stage_openai(
        client,
        stage=RELEVANCY_STAGE,
        tokens={"CATEGORY": category, "ATOMIC_CLAIMS": claims},
        prompts_dir=prompts_dir,
        image_url=image_url,
        single_model=eval_model,
        timeout_s=timeout_s,
    )
    relevancy_record = build_relevancy_record(
        job=relevancy_job,
        image_reference=image_reference,
        model=eval_model,
        result=relevancy_result,
        processed_at=utc_now_iso(),
    )
    if relevancy_record.get("status") != "ok":
        raise RuntimeError(relevancy_record.get("error") or "Relevancy evaluation failed")

    redundancy_job = RedundancyJob(
        algorithm="ViGALT",
        product_stem=product_stem,
        category=category,
        input_claims=_claims_with_relevance(relevancy_record),
        product_dom=product_dom,
        metadata={"final_alt_text": final_alt_text},
    )
    redundancy_result = run_stage_openai(
        client,
        stage=REDUNDANCY_STAGE,
        tokens={
            "PRODUCT_DOM": product_dom,
            "ATOMIC_CLAIMS": redundancy_job.input_claims,
        },
        prompts_dir=prompts_dir,
        image_url="",
        single_model=eval_model,
        timeout_s=timeout_s,
    )
    redundancy_record = build_redundancy_record(
        job=redundancy_job,
        model=eval_model,
        result=redundancy_result,
        processed_at=utc_now_iso(),
    )
    if redundancy_record.get("status") != "ok":
        raise RuntimeError(redundancy_record.get("error") or "Redundancy evaluation failed")

    objectivity_job = ObjectivityJob(
        algorithm="ViGALT",
        product_stem=product_stem,
        category=category,
        alt_text=final_alt_text,
        metadata={},
    )
    objectivity_result = run_stage_openai(
        client,
        stage=OBJECTIVITY_STAGE,
        tokens={"GENERATED_ALT_TEXT": final_alt_text},
        prompts_dir=prompts_dir,
        image_url=image_url,
        single_model=eval_model,
        timeout_s=timeout_s,
    )
    objectivity_record = build_objectivity_record(
        job=objectivity_job,
        image_reference=image_reference,
        model=eval_model,
        result=objectivity_result,
        processed_at=utc_now_iso(),
    )
    if objectivity_record.get("status") != "ok":
        raise RuntimeError(objectivity_record.get("error") or "Objectivity evaluation failed")

    relevant_novel_claims = (
        redundancy_record.get("evaluation", {}).get("summary", {}).get("relevant_novel_claims", 0)
    )
    word_count = count_alt_words(final_alt_text)
    efficiency_score = calculate_efficiency(relevant_novel_claims, word_count)
    efficiency_record = {
        "status": "ok",
        "algorithm": "ViGALT",
        "product_stem": product_stem,
        "category": category,
        "alt_text": final_alt_text,
        "word_count": word_count,
        "relevant_novel_claims": relevant_novel_claims,
        "efficiency_per_100_words": efficiency_score,
        "processed_at": utc_now_iso(),
    }

    return {
        "relevancy": relevancy_record,
        "redundancy": redundancy_record,
        "objectivity": objectivity_record,
        "efficiency": efficiency_record,
    }


def _resolve_image(images_dir: Path, products_dir: Path, product_stem: str) -> tuple[str, str]:
    for ext in IMAGE_EXTENSIONS:
        path = images_dir / f"{IMAGE_BASENAME}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return image_to_data_url(path), str(path.resolve())
    for ext in IMAGE_EXTENSIONS:
        path = images_dir / f"{product_stem}{ext}"
        if path.exists() and path.stat().st_size > 0:
            return image_to_data_url(path), str(path.resolve())
    raise FileNotFoundError(f"No evaluation image found in {images_dir}")


def _claims_with_relevance(relevancy_record: dict) -> list[dict]:
    evaluation = relevancy_record["evaluation"]
    claims = evaluation.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Relevancy evaluation is missing claims.")
    return [
        {
            "claim_id": claim["claim_id"],
            "claim": claim["claim"],
            "relevance_code": claim["label"],
        }
        for claim in claims
    ]


def format_ablation_evaluation(
    *,
    folder_number: int,
    image_file: str,
    variant: AblationVariant,
    final_alt_text: str,
    eval_records: dict,
) -> dict:
    relevancy_eval = eval_records["relevancy"]["evaluation"]
    redundancy_eval = eval_records["redundancy"]["evaluation"]
    objectivity_eval = eval_records["objectivity"]["evaluation"]
    efficiency = eval_records["efficiency"]

    relevancy_claims = relevancy_eval.get("claims", [])
    redundancy_claims = redundancy_eval.get("claims", [])
    objectivity_summary = objectivity_eval.get("summary", {})
    objectivity_claims = objectivity_eval.get("claims", [])

    method_eval = {
        "alt_text": final_alt_text,
        "relevancy": build_relevancy_block(relevancy_claims),
        "redundancy": {
            "metrics": redundancy_eval.get("summary", {}),
            "input_claims": eval_records["redundancy"].get("input_claims", []),
            "claims": redundancy_claims,
        },
        "objectivity": {
            "objectivity_score": objectivity_summary.get("objectivity_score"),
            "objective_claims": objectivity_summary.get("objective_claims"),
            "subjective_claims": objectivity_summary.get("subjective_claims"),
            "total_claims": objectivity_summary.get("total_claims"),
            "claims": objectivity_claims,
        },
        "efficiency": {
            "word_count": efficiency.get("word_count"),
            "relevant_novel_claims": efficiency.get("relevant_novel_claims"),
            "efficiency_per_100_words": efficiency.get("efficiency_per_100_words"),
        },
    }

    return {
        "status": "ok",
        "folder_number": folder_number,
        "image_file": image_file,
        "ablation": variant,
        "models": {
            "fuser": eval_records.get("fuser_model"),
            "claim_extraction": eval_records.get("claim_model"),
            "evaluators": eval_records.get("eval_model"),
        },
        "methods": {"ViGALT_ablation": method_eval},
        "processed_at": utc_now_iso(),
    }


def is_complete(output_dir: Path) -> bool:
    evaluation_path = output_dir / "evaluation.json"
    if not evaluation_path.exists():
        return False
    try:
        record = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return record.get("status") == "ok"


def build_ablations_summary(results: list[dict]) -> dict:
    summary: dict = {
        "generated_at": utc_now_iso(),
        "total_jobs": len(results),
        "variants": {},
    }

    for variant in ABLATION_VARIANTS:
        variant_results = [r for r in results if r.get("variant") == variant]
        ok_results = [r for r in variant_results if r.get("status") == "ok"]
        failed_results = [r for r in variant_results if r.get("status") == "failed"]
        skipped_results = [r for r in variant_results if r.get("status") == "skipped"]

        metric_values: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
        for result in ok_results:
            evaluation_path = Path(result["output_dir"]) / "evaluation.json"
            if not evaluation_path.exists():
                continue
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            method_eval = evaluation["methods"]["ViGALT_ablation"]
            for metric in SUMMARY_METRICS:
                value = metric_value(method_eval, metric)
                if value is not None:
                    metric_values[metric].append(float(value))

        summary["variants"][variant] = {
            "total": len(variant_results),
            "ok": len(ok_results),
            "failed": len(failed_results),
            "skipped": len(skipped_results),
            **{f"avg_{metric}": average(metric_values[metric]) for metric in SUMMARY_METRICS},
        }

    summary["overall"] = {
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
    }
    return summary


def run_single_ablation(
    *,
    client: OpenAI,
    job: AblationJob,
    prompts_dir: Path,
    fuser_model: str,
    claim_model: str,
    eval_model: str,
    timeout_s: float,
    skip_existing: bool,
) -> dict:
    result: dict = {
        "product_id": job.product_id,
        "variant": job.variant,
        "output_dir": str(job.output_dir),
        "status": "failed",
    }

    if skip_existing and is_complete(job.output_dir):
        result["status"] = "skipped"
        return result

    job.output_dir.mkdir(parents=True, exist_ok=True)
    products_dir = job.dataset_dir / str(job.product_id)

    try:
        inputs = load_validation_inputs(job.validations_dir, job.product_id)
        category = category_for_product(job.product_id)
        image_file = inputs["image_path"].name

        fusion_record = run_fusion(
            client=client,
            variant=job.variant,
            inputs=inputs,
            prompts_dir=prompts_dir,
            fuser_model=fuser_model,
            timeout_s=timeout_s,
        )
        save_output(job.output_dir / "fusion.json", fusion_record)

        claims_record = run_claim_extraction(
            client=client,
            final_alt_text=fusion_record["final_alt_text"],
            product_id=job.product_id,
            category=category,
            prompts_dir=prompts_dir,
            claim_model=claim_model,
            timeout_s=timeout_s,
        )
        save_output(job.output_dir / "claims.json", claims_record)

        eval_image_dir = job.output_dir
        ensure_eval_image(inputs["image_path"], eval_image_dir)

        eval_records = run_evaluations(
            client=client,
            job=job,
            claims_record=claims_record,
            final_alt_text=fusion_record["final_alt_text"],
            eval_image_dir=eval_image_dir,
            products_dir=products_dir,
            prompts_dir=prompts_dir,
            eval_model=eval_model,
            timeout_s=timeout_s,
        )
        eval_records["fuser_model"] = fuser_model
        eval_records["claim_model"] = claim_model
        eval_records["eval_model"] = eval_model

        evaluation = format_ablation_evaluation(
            folder_number=job.product_id,
            image_file=image_file,
            variant=job.variant,
            final_alt_text=fusion_record["final_alt_text"],
            eval_records=eval_records,
        )
        save_output(job.output_dir / "evaluation.json", evaluation)
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = str(exc)
        save_output(
            job.output_dir / "error.json",
            {
                "status": "failed",
                "product_id": job.product_id,
                "variant": job.variant,
                "error": str(exc),
                "processed_at": utc_now_iso(),
            },
        )

    return result


def run_ablation_batch(
    *,
    validations_dir: Path,
    dataset_dir: Path,
    product_ids: list[int],
    variants: list[AblationVariant],
    fuser_model: str,
    claim_model: str,
    eval_model: str,
    timeout_s: float,
    delay_s: float,
    skip_existing: bool,
    client: OpenAI,
) -> dict:
    prompts_dir = project_root() / PROMPTS_DIR
    jobs: list[AblationJob] = []

    for product_id in product_ids:
        for variant in variants:
            output_dir = dataset_dir / str(product_id) / "ablations" / variant
            jobs.append(
                AblationJob(
                    product_id=product_id,
                    variant=variant,
                    validations_dir=validations_dir,
                    dataset_dir=dataset_dir,
                    output_dir=output_dir,
                )
            )

    results: list[dict] = []
    total = len(jobs)

    print("=" * 60)
    print("Validator Ablation Study")
    print("=" * 60)
    print(f"Products:  {len(product_ids)}")
    print(f"Variants:  {', '.join(variants)}")
    print(f"Jobs:      {total}")
    print(f"Fuser:     {fuser_model}")
    print(f"Claims:    {claim_model}")
    print(f"Eval:      {eval_model}")
    print()

    for index, job in enumerate(jobs, start=1):
        label = f"[{index}/{total}] product {job.product_id} / {job.variant}"
        print(label)
        result = run_single_ablation(
            client=client,
            job=job,
            prompts_dir=prompts_dir,
            fuser_model=fuser_model,
            claim_model=claim_model,
            eval_model=eval_model,
            timeout_s=timeout_s,
            skip_existing=skip_existing,
        )
        results.append(result)
        print(f"  -> {result['status']}")
        if result.get("error"):
            print(f"  error: {result['error']}")
        if index < total and delay_s > 0:
            time.sleep(delay_s)

    summary = build_ablations_summary(results)
    summary_path = dataset_dir / "ablations_summary.json"
    save_output(summary_path, summary)
    print()
    print(f"Summary written to: {summary_path}")
    return summary
