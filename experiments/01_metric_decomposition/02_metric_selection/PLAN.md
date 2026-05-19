# 02_metric_selection — Documenting the Primary Evaluation Metric Choice

## Purpose

This is a **confirmation appendix**, not a discovery experiment. The answer is known
in advance: per-drug Pearson r is the right primary metric. This experiment documents
the empirical basis for that choice so it can be cited in the paper methods section.

The expected findings:
- Pearson r and Spearman r are interchangeable at n ≈ 22 cells/drug (ρ > 0.95)
- R² collapses to Pearson² in per-drug evaluation (drug mean is always the baseline)
- Kendall τ is redundant with Spearman r
- NDCG@5 is unstable at k=5/n=22 — failure mode documented for the record
- Pearson r is preferred over Spearman r for community comparability

If the bootstrap CIs contradict the Pearson-Spearman equivalence (CI widths differ
substantially), the conclusion updates accordingly. That is the only live question.

## Context

`01_global_vs_perdrug` established that global Pearson r is broken: 68.1% of IC₅₀
variance is between drugs, Predictor 2A achieves global r = 0.837 with per-drug r = 0,
and model rankings reverse across splits. Sealfon et al. (J. Cheminformatics 2025)
independently arrived at the same conclusion and endorsed Fixed-Drug Pearson r (= our
per-drug Pearson r) as the replacement. This experiment provides the additional evidence
needed to choose *which* per-drug metric, a question Sealfon et al. did not address.

## Candidate metrics

| # | Metric | Symbol | Expected result |
|---|--------|--------|-----------------|
| 1 | Per-drug Pearson r | r_p | Primary metric — stable, interpretable, literature-standard |
| 2 | Per-drug Spearman r | r_s | Interchangeable with r_p at n ≈ 22 |
| 3 | Per-drug Kendall τ | τ | Redundant with r_s |
| 4 | Per-drug NDCG@5 | NDCG | Unstable at k=5/n=22; failure documented |
| 5 | Per-drug R² | R² | Collapses to r_p² in per-drug evaluation |

Global Pearson r is excluded — its failure was already established in 01.
Predictor 2A (drug-mean constant) is included as a sanity check only.

### Metric implementation specs

**R²**: Custom per-drug R², defined as `1 - SS_res / SS_tot` where `SS_tot` uses the
per-drug mean of y_true (not the global test-set mean). This makes R² invariant to
drug-mean shifts and gives the r_p² relationship. *sklearn's `r2_score` uses the
test-set mean as baseline — do not use it directly.*

**NDCG@5**: `sklearn.metrics.ndcg_score` with k=5. On Predictor 2A (all predictions
equal within a drug), sklearn treats all items as tied and returns a value near the
maximum (not 0.0). Criterion (a) therefore does not apply to NDCG as a 0.0 test;
instead, NDCG on Predictor 2A is recorded and reported as-is.

**Drug filter**: Before any metric computation for a given (drug, fold) pair, check
`std(y_true) > 1e-6`. Drugs that fail this check are excluded from that fold (constant
response panel → Pearson/Spearman/τ undefined). Report excluded count per fold.

**Ridge hyperparameter**: Nested CV over α ∈ {0.01, 0.1, 1, 10, 100, 1000},
selected by global Pearson r on the validation fold — matching the alpha grid in
`01_global_vs_perdrug/src/runner.py`. Per-fold best α is logged.

## Evaluation criteria

### (a) Sanity check — Predictor 2A

All per-drug metrics must score 0.0 on a constant-within-drug predictor. This is a
pass/fail check, not a discriminating criterion. Any metric that fails is misimplemented.

### (b) Statistical stability — bootstrap CI width at n ≈ 22

Bootstrap 200 resamples per (drug, fold) pair. Compute 95% CI width per metric,
aggregate (median, IQR) across drugs and folds. Report the Pearson-Spearman CI width
ratio (median r_p CI width / median r_s CI width).

**Decision rule** (discussion threshold, not a formal test):
- Ratio > 1.1: Spearman is meaningfully more stable → primary metric is r_s
- Ratio ≤ 1.1: metrics are equivalent → primary metric is r_p (community comparability)

The 1.1 value is not principled — it reflects "10% wider CI is practically relevant at
n ≈ 22 when averaging over ≈ 22 drugs per fold." The actual ratio is the result; the
threshold converts it to a recommendation. Report the raw ratio regardless.

PMC 2025 notes per-drug Pearson r instability at n ≈ 20–30 but does not quantify it.
This closes that gap for the five candidate metrics.

### (c) Inter-metric correlation

Spearman correlation between all metric pairs across (drug, fold) entries.
- r_p vs r_s ≥ 0.95: interchangeable → retain r_p (interpretability)
- r_s vs τ ≥ 0.95: τ is redundant → drop τ
- r_p vs R²: expected to reveal the r_p² structure
- r_p vs NDCG: expected low → documents NDCG as measuring a different quantity

The 0.95 threshold is a discussion threshold, not a formal test.

## Prior work on metric selection

**What exists and what it covers:**

| Paper | Venue | Relevant claim |
|-------|-------|----------------|
| Sealfon et al. "The specification game" | J. Cheminformatics 17:28, 2025 | DummyDrugAvg achieves global r = 0.85; Fixed-Drug Pearson r (= per-drug Pearson r) collapses to 0.00 on 2A; endorses per-drug aggregation strategies. Does NOT compare Spearman, Kendall, NDCG, or R². Does NOT report bootstrap CI widths. |
| PMC 2025 methodological review | PMC (exact citation in paper bib) | Notes per-drug Pearson r is unstable at n ≈ 20–30 cells/drug; recommends CI reporting or bootstrap aggregation. Does NOT quantify instability empirically or compare competing metrics. |
| Partin et al. | Briefings in Bioinformatics, 2026 (arXiv 2503.14356) | Argues R² preferred for cross-dataset generalisation; drug-set coverage is the main driver of out-of-distribution performance. R² is used without per-drug mean correction — differs from our custom R² definition. |
| Fröhlich et al. DrEval | bioRxiv, May 2025 | Enumerates six failure modes of standard DRP evaluation including metric conflation. Benchmarks multiple models but uses global metrics, not per-drug. |
| Hafner et al. | Nature Methods, 2016 | GR metrics (normalised growth-rate inhibition) expose cell-line-specific effects masked by absolute IC₅₀; foundational critique of using un-normalised response as target. |
| Ovchinnikova et al. | npj Precision Oncology, 2024 | Per-drug z-scoring of IC₅₀ recommended before any evaluation; global metrics conflate inter-drug variance. |

**Gap this experiment fills:** No prior paper has (a) formally compared the bootstrap CI widths of Pearson, Spearman, Kendall τ, NDCG@k, and R² at n ≈ 22 cells/drug, or (b) reported their pairwise inter-metric Spearman correlations across real DRP data. Sealfon endorses Fixed-Drug Pearson r but provides no empirical comparison against Spearman.

## Implementation plan

Steps 1 and 2 are parallelised across folds via SLURM array jobs.
Step 3 runs after all fold jobs complete. Submit with `jobs/launch.sh`.

```
Step 1 (array 0-4)     Step 1 (array 0-4)     ...   — 5 Ridge fold jobs, ~2 min each
         ↓ afterok
Step 2 (array 0-4)     Step 2 (array 0-4)     ...   — 5 bootstrap jobs, ~1 min each
         ↓ afterok (all)
Step 3 (single)                                      — aggregate, < 30 sec
```

### Step 1 — Save Ridge fold predictions  `step1_ridge.sbatch` (array 0-4)

Each array task runs `save_ridge_predictions.py --fold $SLURM_ARRAY_TASK_ID`.
Saves raw (depmap_id, drug_name, y_true, y_pred) for that fold.

Output: `results/fold_predictions/ridge_drug_blind_fold{0..4}.parquet`

### Step 2 — Per-drug metrics and bootstrap  `step2_bootstrap.sbatch` (array 0-4)

Each array task runs `compute_fold_metrics.py --fold $SLURM_ARRAY_TASK_ID`.
Computes r_p, r_s, τ, NDCG@5, R² per (drug, fold), plus Predictor 2A sanity
columns. Then bootstraps 200× per drug to get CI widths.

Output per fold:
- `results/fold_metrics/fold{i}_per_drug.parquet`
- `results/fold_metrics/fold{i}_bootstrap.json`

### Step 3 — Aggregate  `step3_aggregate.sbatch` (single job)

Runs `aggregate_results.py`. Concatenates fold parquets, aggregates CI widths
(median, IQR across all folds), computes pairwise Spearman correlation across
all (drug, fold) entries. Logs the recommendation.

Output:
- `results/per_drug_metrics.parquet`
- `results/metric_analysis.json`

```json
{
  "ci_width": {
    "r_p": {"median": ..., "iqr": ..., "n_drugs_folds": ...},
    ...
  },
  "pearson_spearman_ci_ratio": ...,
  "inter_metric_spearman": {"r_p_vs_r_s": ..., "r_s_vs_tau": ..., ...},
  "n_drugs_per_fold": {"0": ..., "1": ..., ...}
}
```

### Step 4 — Report  (manual)

One-page `report/README.md` stating:
1. Sanity check results (pass/fail table for Predictor 2A)
2. CI width table with Pearson/Spearman ratio
3. Inter-metric correlation matrix (5×5)
4. Recommendation with rationale

**Paper methods sentence — Branch A (expected, ratio ≤ 1.1):**
"We use per-drug Pearson r as the primary metric. Bootstrap analysis at n ≈ 22
cells/drug confirms Pearson r and Spearman r are interchangeable (ρ = X,
CI width ratio = Y ≤ 1.1); Pearson r is preferred for comparability with
existing DRP literature."

**Paper methods sentence — Branch B (if ratio > 1.1):**
"We use per-drug Spearman r as the primary metric. Bootstrap analysis at n ≈ 22
cells/drug shows Spearman r has meaningfully narrower confidence intervals than
Pearson r (CI width ratio = Y > 1.1), indicating greater stability at our sample
size. Per-drug Pearson r is reported alongside for comparability."

## Files

```
02_metric_selection/
  PLAN.md
  src/
    save_ridge_predictions.py     ← Step 1 (--fold arg or SLURM_ARRAY_TASK_ID)
    compute_fold_metrics.py       ← Step 2 (--fold arg or SLURM_ARRAY_TASK_ID)
    aggregate_results.py          ← Step 3
  jobs/
    launch.sh                     ← submit all three steps with dependencies
    step1_ridge.sbatch            ← array 0-4, ~2 min/task
    step2_bootstrap.sbatch        ← array 0-4, ~1 min/task, depends on step1
    step3_aggregate.sbatch        ← single job, < 30 sec, depends on step2
  results/
    fold_predictions/             ← Step 1 output
    fold_metrics/                 ← Step 2 output (per-fold intermediates)
    per_drug_metrics.parquet      ← Step 3 output
    metric_analysis.json          ← Step 3 output
  report/
    README.md                     ← Step 4 output
```
