## Fusion-stage validator ablation

<!-- generated: 2026-09-28T09:46:45.447624+00:00; regenerate with python -m viglat.build_ablation_report -->

We isolate the contribution of each parallel validator at **fusion** by re-compiling alt text from fixed validator outputs while omitting one validator channel (`no_accuracy`, `no_completeness`, `no_redundancy`). Validators still run; only the fuser input changes. The ablation baseline is full ViGALT (`evaluation_dataset/N/evaluation.json`), not the ASSETS24 comparison in the main evaluation.

### Setup

- **n = 30** products (clothing 1–15, furniture 16–30); **90** ablation runs (3 variants × 30), all `status: ok`.
- **Held constant:** `ViGALT_validations/N/` artifacts (atomic claims, accuracy/completeness/redundancy JSON), image, and DOM.
- **Varied:** fuser prompt — `fuser_no_accuracy.txt`, `fuser_no_completeness.txt`, or `fuser_no_redundancy.txt` ([`src/viglat/evaluation/ablation.py`](src/viglat/evaluation/ablation.py)).
- **Primary evidence:** validator-native structural outcomes (claims added/lost, corrections/deletions). Post-hoc DOM metrics on ablated outputs are reported separately and are not used to claim overall system superiority.

### Results by validator

#### Completeness (visual coverage)

| Metric | Value |
|---|---:|
| Products with ≥1 completeness `ADDED` claim | 27 / 30 |
| Total completeness `ADDED` claims | 48 |
| Mean `ADDED` claims per product | 1.60 |
| `ADDED` claims absent under `no_completeness` | 48 / 48 (100.0%) |
| Mean Δ claims (no_completeness − baseline) | -1.37 |
| Mean Δ words (no_completeness − baseline) | -9.47 |
| Products with fewer claims under `no_completeness` | 24 / 30 |
| Clothing mean Δ claims / Δ words | -1.67 / -10.93 |
| Furniture mean Δ claims / Δ words | -1.07 / -8.00 |

Without completeness at fusion, **all** completeness `ADDED` claims are excluded from the compiled alt text; outputs are shorter and contain fewer claims.

#### Accuracy (correction and deletion authority)

| Metric | Value |
|---|---:|
| Products with `CORRECTED` or `DELETED` claims | 7 / 30 |
| Total `CORRECTED` claims | 8 |
| Total `DELETED` claims | 1 |

Affected products (baseline vs `no_accuracy` alt text differs on all listed cases):

| Product | Interventions |
|---:|---|
| 5 | CORRECTED `Sweatshirt` → `Zip-up hooded sweatshirt` |
| 6 | CORRECTED `Black contrasting collar` → `Gray collar with a black inner collar` |
| 14 | CORRECTED `Two-piece pajama set` → `Two-piece outfit consisting of a shirt and shorts` |
| 22 | DELETED `A sofa bed is arranged as a single bed` |
| 27 | CORRECTED `The mirror has black plastic borders.` → `The mirror has black borders.`; CORRECTED `Two black over-door hooks extend above the mirror.` → `Two black hooks extend above the mirror.` |
| 28 | CORRECTED `The mirror has straight vertical sides.` → `The mirror has straight, slightly angled side edges.` |
| 30 | CORRECTED `Round bathroom mirror` → `Round mirror`; CORRECTED `Three touch-control icons near the lower center of the reflective surface` → `Three icons near the lower center of the reflective surface` |

Accuracy interventions are **sparse** (7/30 products) but **decisive** where they occur: without accuracy at fusion, corrected wording can revert and one deleted claim (product 22) re-enters the alt text.

#### Redundancy (DOM repetition control)

| Metric | Value |
|---|---:|
| Products with pipeline `AVOIDABLE_REDUNDANCY` labels | 25 / 30 |
| Total pipeline `AVOIDABLE_REDUNDANCY` labels | 28 |
| Mean Δ avoidable redundancy rate (`no_redundancy` − baseline) | 0.048 |
| Mean Δ claims / Δ words under `no_redundancy` | 0.60 / 0.60 |
| Products with increased avoidable redundancy (Δ > 0.01) | 14 / 30 |

Redundancy has a **moderate** aggregate effect: removing it at fusion retains more avoidable DOM repetition in 14/30 products, with smaller average shifts in length and claim count than completeness.

### Post-hoc metrics on ablated outputs

The table below re-evaluates ablated alt texts with the same relevancy/redundancy/objectivity/efficiency pipeline used elsewhere. Ablated variants often score **higher** on efficiency/novelty and **lower** on redundancy because validators intentionally add coverage and corrected wording—this tension is expected and does **not** imply that removing validators improves ViGALT.


| Variant | Relevancy | Objectivity | Efficiency† | Claims | Words | Avoidable redund. | Redundancy | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline ViGALT | 90.10 | 0.977 | 5.94 | 6.9 | 44.7 | 0.178 | 0.590 | 0.361 |
| no_accuracy | 89.90 | 0.990 | 7.09 | 7.0 | 43.9 | 0.163 | 0.510 | 0.428 |
| no_completeness | 87.28 | 0.987 | 7.22 | 5.5 | 35.2 | 0.045 | 0.467 | 0.460 |
| no_redundancy | 89.75 | 0.993 | 7.05 | 7.5 | 45.3 | 0.225 | 0.525 | 0.411 |

† Efficiency = relevant novel claims per 100 words.

### Illustrative cases

#### Completeness — baseline vs `no_completeness`

**Product 1** (completeness `ADDED`: Short sleeves; Crew neckline)

- *Full fusion:* A T-shirt shown from the front, with short sleeves and a crew neckline. A small gray Under Armour logo is on the wearer's upper left chest.
- *no_completeness:* A T-shirt is shown from the front. It has a small gray Under Armour logo on the wearer's upper left chest.

**Product 11** (completeness `ADDED`: Dark navy color; Long sleeves; Ribbed crew neckline, cuffs, and waistband; Colorful liquid streaming from the held planet into the pools)

- *Full fusion:* Dark navy color, long sleeves, and a ribbed crew neckline, cuffs, and waistband are visible. A central space-themed graphic shows a white-outlined astronaut holding a multicolored planet, with colorful liquid streaming into pools beneath. Stars, ringed planets, additional small planets, and a moon-like circle surround the astronaut.
- *no_completeness:* A central space-themed graphic features a white-outlined astronaut holding a multicolored planet, with stars and ringed planets surrounding the astronaut. Colorful pools appear beneath the astronaut, with additional small planets and a moon-like circle above.

**Product 7** (completeness `ADDED`: Long sleeves; Pointed turn-down collar; White buttons down the front)

- *Full fusion:* Intersecting white and dark-blue stripes are visible. Visible features include a pointed turn-down collar, long sleeves, a chest pocket on the wearer’s left with diagonally oriented stripes, and a striped front placket with white buttons down the front. The hem is gently curved.
- *no_completeness:* Intersecting white and dark-blue stripes are present, along with a chest pocket on the wearer’s left with diagonally oriented stripes, a striped front placket, and a gently curved hem.

#### Accuracy — baseline vs `no_accuracy`

**Product 6**

- *Full fusion:* Gray shirt body with a gray collar and a black inner collar. Black buttons are arranged down the center front, with a rounded lower edge and narrow cuffs at the ends of the sleeves. A black cuff is visible at the end of the sleeve on the right side of the image.
- *no_accuracy:* The gray shirt body has a black contrasting collar and black buttons arranged down the center front. Its rounded lower edge and narrow cuffs at the ends of the sleeves are visible, including a black cuff at the end of the sleeve on the right side of the image.

**Product 22**

- *Full fusion:* An L-shaped frame surrounds the head and right side, with open shelving illuminated by cyan LEDs. A work surface extends across the foot and right side, with a laptop sitting on it. Storage compartments beneath the mattress are pulled open on the left, and fabric organizer pockets hang from the near side.
- *no_accuracy:* A sofa bed is arranged as a single bed. An L-shaped frame surrounds the head and right side, with open shelving illuminated by cyan LEDs. The work surface extends across the foot and right side, and a laptop sits on the work surface. Storage compartments beneath the mattress are pulled open on the left, and fabric organizer pockets hang from the near side.

**Product 28**

- *Full fusion:* A mirror has a narrow border and a dark-colored frame, with straight, slightly angled side edges. A separate support extends behind the mirror on the right and holds it upright on the floor
- *no_accuracy:* A mirror with a narrow border, straight vertical sides, and a dark-colored frame. A separate support extends behind it on the right, holding it upright on the floor.

### Ablation limitations

- Fusion-stage only: validator outputs are fixed; we measure fuser dependence, not removing validator stages entirely.
- Accuracy evidence is count- and case-based (8 corrections, 1 deletion in this slice).
- Ablated alt texts are not re-scored by the accuracy or completeness validator prompts.
- Descriptive study; no significance tests.

### LaTeX paste block (optional)

```latex
\begin{table}[t]
\centering
\caption{Fusion-stage validator ablation (n=30). Primary structural outcomes.}
\begin{tabular}{lrr}
\toprule
Validator removed & Primary outcome & Value \\
\midrule
Completeness & ADDED claims lost at fusion & 48/48 (100.0%) \\
Completeness & Mean $\Delta$ claims / words & -1.37 / -9.47 \\
Accuracy & Products with CORRECTED/DELETED & 7/30 \\
Redundancy & Mean $\Delta$ avoidable redundancy & 0.048 \\
\bottomrule
\end{tabular}
\end{table}
```
