# ViGALT — Visual Gap-Aware ALT Text Generation

Research codebase for **ViGALT**, a multi-stage pipeline that generates high-quality, screen-reader-friendly alt text for e-commerce product images. The system takes a product image plus surrounding page text and produces objective, hallucination-filtered alt text through a 7-stage generate → deconstruct → validate → fuse pipeline.

This repository is the public release accompanying our research paper. It includes the full source code, prompts, and the consolidated evaluation dataset with published results for **ViGALT** vs. **CAID** (Mohanbabu and Pavel, ASSETS '24).

## Pipeline Overview

```mermaid
flowchart LR
  classifier[Classifier] --> generator[Generator]
  generator --> altToList[ALT-to-List]
  altToList --> accuracy[Accuracy Validator]
  altToList --> completeness[Completeness Validator]
  altToList --> redundancy[Redundancy Validator]
  accuracy --> fuser[Fuser]
  completeness --> fuser
  redundancy --> fuser
  fuser --> finalAlt[final_alt_text]
```

Every product is classified as **CLOTHING** or **FURNITURE**. Stages 4–6 run in parallel; all other stages are sequential. See [docs/pipeline_architecture.md](docs/pipeline_architecture.md) for the full specification.

## Repository Layout

```
.
├── src/viglat/              Python package (pipeline + evaluators)
├── prompts/               Stage prompts and evaluation rubrics
├── evaluation_dataset/    Published per-image results (30 folders + summary.json)
├── docs/                  Architecture documentation + validator ablation report
├── output/                Generated runs (created when you execute the pipeline)
├── pyproject.toml         Package metadata
├── requirements.txt       Pinned dependencies
└── .env.example           API key template
```

### `src/viglat/` structure

```
src/viglat/
├── paths.py                 Stable project-root resolution
├── pipeline/                7-stage runner, utils, types
├── backend/                 OpenAI client, config, JSON schemas
├── extraction/              Amazon DOM scraper
├── evaluation/              Relevancy, redundancy, objectivity, efficiency evaluators
├── cli/                     extract_dom, run_from_url, build_dataset, combine_claims
└── *.py shims               Backward-compatible entry points (e.g. pipeline via package)
```

| Group | Location |
|---|---|
| Pipeline core | `pipeline/runner.py`, `pipeline/utils.py`, `pipeline/types.py` |
| Backend | `backend/openai.py`, `backend/config.py`, `backend/schemas.py` |
| DOM extraction | `cli/extract_dom.py`, `extraction/amazon.py` |
| Orchestration | `cli/run_from_url.py` |
| Evaluators | `evaluation/relevancy.py`, `redundancy.py`, `objectivity.py`, `efficiency.py`, `alt_to_list.py` |
| Dataset tools | `cli/build_dataset.py`, `cli/combine_claims.py` |

All documented `python -m viglat.*` commands still work via thin shims at the package root.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
pip install -e .
```

Set your OpenAI API key (required for all LLM stages). Either copy `.env.example` to `.env` and fill in your key, or export it in the shell:

```powershell
# Option A: .env file (recommended)
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# Option B: PowerShell (Windows)
$env:OPENAI_API_KEY = "sk-..."
```

```bash
# Option A: .env file (recommended)
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# Option B: macOS/Linux
export OPENAI_API_KEY=sk-...
```

The `.env` file in the project root is loaded automatically when you run any `python -m viglat.*` command.

Default pipeline model is **`gpt-5.6-luna`**; evaluation stages (relevancy, redundancy, objectivity) use **`gpt-5.6-terra`** (configured in [`src/viglat/backend/config.py`](src/viglat/backend/config.py)). Override with `--single-model` or `--model` on any command.

---

## Quick start: one Amazon URL → full run

Runs DOM extraction, alt-text generation, claim export, and all four evaluation metrics:

```bash
python -m viglat.run_from_url --url "https://www.amazon.fr/dp/B077XM3DV5"
```

Each run is written to a numbered folder with fixed filenames:

```
output/runs/
  1/
    dom.json
    image.jpg
    meta.json
    pipeline.json
    claims.json
    relevancy.json
    redundancy.json
    objectivity.json
    efficiency.json
  2/
    ...
```

The next free run number is chosen automatically (`1`, then `2`, …).

Options:

```bash
python -m viglat.run_from_url --url "..." --run-id 5           # use output/runs/5/
python -m viglat.run_from_url --url "..." --work-dir output/runs/custom
python -m viglat.run_from_url --html saved_page.html --work-dir output/runs/3
python -m viglat.run_from_url --url "..." --skip-eval          # pipeline only
python -m viglat.run_from_url --url "..." --single-model gpt-5.6-luna
```

---

## Step-by-step guide

### Step 1 — Extract product DOM + image

Build a product JSON from an Amazon URL. Downloads the main product image into the same folder.

```bash
python -m viglat.extract_dom \
  --url "https://www.amazon.fr/dp/B077XM3DV5" \
  --output-dir output/runs/1
```

Creates:
- `output/runs/1/dom.json` — title, brand, description, feature bullets, product details, `main_image`
- `output/runs/1/image.jpg` — downloaded product image (single file, no duplicate)
- `output/runs/1/meta.json` — run id, ASIN, source URL

If Amazon blocks automated requests, save the page HTML in your browser and parse locally:

```bash
python -m viglat.extract_dom --html saved_page.html --output-dir output/runs/1
```

You can also use a product JSON from the paper dataset: `evaluation_dataset/1/dom.json` (each folder `1/` … `30/` contains metadata for one product).

### Step 2 — Generate alt text (7 pipeline stages)

One command runs all generation stages via OpenAI:

| Order | Stage | Prompt | Input |
|---|---|---|---|
| 1 | Classifier | `prompts/classifier.txt` | image + surrounding text |
| 2 | Generator | `prompts/clothing_generator.txt` or `furniture_generator.txt` | image + surrounding text |
| 3 | Claim extraction | `prompts/alt_to_list.txt` | generated alt text |
| 4 | Accuracy validator | `prompts/accuracy_validator.txt` | image + claims |
| 5 | Completeness validator | `prompts/completeness_validator.txt` | image + claims |
| 6 | Redundancy validator | `prompts/redundancy_validator.txt` | image + claims |
| 7 | Fuser | `prompts/fuser.txt` | validator outputs |

```bash
python -m viglat.pipeline --product output/runs/1/dom.json --output output/runs/1/pipeline.json
```

Or with a paper-dataset product:

```bash
python -m viglat.pipeline --product evaluation_dataset/1/dom.json
```

Output: `output/<stem>_pipeline.json` with `final_alt_text` and per-stage results.

### Step 3 — Extract claims (evaluation format)

The pipeline already runs claim extraction inline (stage 3). This step re-extracts claims into the batch format used by evaluators:

```bash
python -m viglat.alt_to_list_batch \
  --source pipeline \
  --pipeline output/B077XM3DV5_pipeline.json \
  --output-dir output/runs/B077XM3DV5/final_alt_claim_lists
```

Output: `output/runs/B077XM3DV5/final_alt_claim_lists/B077XM3DV5_claims.json`

### Step 4 — Relevancy evaluation

```bash
python -m viglat.relevancy_evaluator_batch \
  --algorithm ViGALT \
  --vigalt-claims-dir output/runs/B077XM3DV5/final_alt_claim_lists \
  --products-dir output/runs/B077XM3DV5 \
  --images-dir output/runs/B077XM3DV5 \
  --output output/runs/B077XM3DV5/relevancy_evaluations.json
```

Prompt: `prompts/evaluators/relevancy.txt`

### Step 5 — Redundancy evaluation

```bash
python -m viglat.redundancy_evaluator_batch \
  --algorithm ViGALT \
  --relevancy-input output/runs/B077XM3DV5/relevancy_evaluations.json \
  --products-dir output/runs/B077XM3DV5 \
  --output output/runs/B077XM3DV5/redundancy_evaluations.json
```

Prompt: `prompts/evaluators/redundancy.txt`

### Step 6 — Objectivity evaluation

```bash
python -m viglat.objectivity_evaluator_batch \
  --algorithm ViGALT \
  --relevancy-input output/runs/B077XM3DV5/relevancy_evaluations.json \
  --products-dir output/runs/B077XM3DV5 \
  --images-dir output/runs/B077XM3DV5 \
  --output output/runs/B077XM3DV5/objectivity_evaluations.json
```

Prompt: `prompts/evaluators/objectivity.txt`

### Step 7 — Efficiency evaluation

Deterministic metric (no LLM): `(relevant_novel_claims / ALT word count) × 100`

```bash
python -m viglat.efficiency_evaluator_batch \
  --algorithm ViGALT \
  --redundancy-input output/runs/B077XM3DV5/redundancy_evaluations.json \
  --relevancy-input output/runs/B077XM3DV5/relevancy_evaluations.json \
  --output output/runs/B077XM3DV5/efficiency_evaluations.json
```

---

## Paper dataset (30 products)

### What is included

The `evaluation_dataset/` folder contains everything needed to **verify** the paper numbers:

| File per folder (`1/` … `30/`) | Contents |
|---|---|
| `dom.json` | Product metadata (title, brand, features, `main_image` URL, …) |
| `alt_text_and_claims.txt` | ViGALT and CAID alt text + plain claims |
| `evaluation.json` | Relevancy, redundancy, objectivity, efficiency metrics |

Top-level `evaluation_dataset/summary.json` holds macro-averaged metrics across all 30 products (each product weighted equally).

- **30 products:** folders `1/` … `30/` (15 clothing, 15 furniture)
- Product JSONs live at `evaluation_dataset/N/dom.json` — not in a separate `data/` folder
- Images are referenced by URL in `dom.json`; the pipeline downloads them on first run

### Re-run pipeline on a paper product

```bash
python -m viglat.pipeline --product evaluation_dataset/1/dom.json
python -m viglat.pipeline --product evaluation_dataset/1/dom.json --single-model gpt-5.6-luna
```

To process all 30 products, run the pipeline once per folder (or use a shell loop over `evaluation_dataset/*/dom.json`).

### Verify published metrics

Compare your outputs to the pre-computed files in `evaluation_dataset/N/evaluation.json` and the averages in `evaluation_dataset/summary.json`. No re-run is required to check the paper numbers.

Recompute relevancy scores from existing R/S/V claim labels (same formula as `prompts/evaluators/relevancy.txt`) and refresh summaries:

```bash
python -m viglat.refresh_relevancy_metrics
```

Regenerate paper figures (matplotlib; requires `pip install -e ".[viz]"`):

```bash
python -m viglat.generate_paper_figures
```

### Validator ablation (fusion stage)

Run fusion without one validator input (requires prior `ViGALT_validations/` artifacts):

```bash
python -m viglat.run_validator_ablation --products 1-30 --skip-existing
```

Regenerate the ablation study section (for insertion into your main report) and machine-readable aggregates:

```bash
python -m viglat.build_ablation_report
```

Outputs:

- [`docs/validator_ablation_report.md`](docs/validator_ablation_report.md) — fusion-stage ablation only (setup, results, illustrative cases)
- [`evaluation_dataset/ablation_summary.json`](evaluation_dataset/ablation_summary.json) — reproducible summary statistics
- Per-product raw ablation artifacts: `evaluation_dataset/N/ablations/{no_accuracy,no_completeness,no_redundancy}/`

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Relevancy | Per product: classify each claim as R, S, or V, then score with `100 × (R + 0.5·S) / (R + S + 2·V)` (see `prompts/evaluators/relevancy.txt`). `evaluation.json` stores `relevancy_score` (0–100); `summary.json` stores `avg_relevancy_score` on a 0–1 scale (macro mean over 30 products). |
| Redundancy | Novel vs. avoidable repetition relative to product DOM |
| Objectivity | Objective vs. subjective claim labels |
| Efficiency | `(relevant_novel_claims / ALT word count) × 100` |

## Notes

- Amazon DOM extraction may be blocked by bot detection; use `--html` with a saved page when needed. Scraping is your responsibility under Amazon's terms of service.
- Re-running the pipeline produces new alt texts; LLM outputs vary. Use `evaluation_dataset/` to verify published paper numbers.
- If you see `viglat.egg-info/` under `src/` after `pip install -e .`, that is a local build artifact (gitignored).

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{vigalt2026,
  title   = {ViGALT: Visual Gap-Aware ALT Text Generation for E-Commerce Product Images},
  author  = {TODO: Add authors},
  journal = {TODO: Add venue},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
