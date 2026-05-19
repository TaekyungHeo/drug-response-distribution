# 02_active_selection — Which K cells to screen for maximum K-shot gain?

## Research question

Given a budget of K cells to screen for a new drug, which cell selection
strategy maximizes the per-drug prediction gain from response matching
(01_response_matching)?

Random selection is the default in 01_response_matching. But cells differ in
informativeness: high-variance cells discriminate between drugs, while
redundant cells waste the budget.

## Hypothesis

**Null**: Cell selection strategy does not affect K-shot per-drug r. Random
selection is as good as any targeted strategy at all K.

**Alternative**: Targeted selection outperforms random, with the optimal
strategy depending on K:
- K <= 5: MaxVar (high inter-drug variance cells) dominates, because each
  observation must maximally discriminate between candidate drug profiles.
- K >= 10: MidResp (cells near median response) dominates, because the budget
  is large enough that calibration accuracy matters more than discrimination.

Expected magnitudes (at K=10):
- Random: per-drug r ~ 0.68 (from 01_response_matching)
- MaxVar: per-drug r ~ 0.71
- MidResp: per-drug r ~ 0.70
- Diverse: per-drug r ~ 0.69

## Design

**Data**: GDSC2, 233 drugs, 687 cell lines, PASO 10-fold drug-blind CV.

**Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200).

**Cell selection strategies**:
1. **Random**: uniform random sample of K cells (baseline, matches
   01_response_matching).
2. **MaxVar**: select the K cells with highest inter-drug IC50 variance across
   training drugs. These cells show the widest spread in sensitivity, making
   them most discriminative for identifying similar drugs.
3. **MidResp**: select the K cells whose mean IC50 (across training drugs) is
   closest to the global median. These are calibration points that anchor the
   drug's absolute potency without being confounded by extreme sensitivity
   or resistance.
4. **Diverse**: select K cells that are maximally spread in RNA feature space.
   Greedy farthest-first traversal in PCA(550) space. These cells cover
   the biological diversity of the panel, avoiding redundant cell types.

**Procedure**:
1. For each fold:
   a. Compute cell statistics from training drugs only (variance, mean response,
      RNA PCA coordinates). This avoids leaking test drug information.
   b. For each strategy, select K cells.
   c. For each test drug, run the response-matching pipeline from
      01_response_matching using the selected K cells as anchors.
   d. Compute per-drug Pearson r on non-anchor cells.
2. Average per-drug r across folds.

**K values**: 1, 3, 5, 10, 20.

**Blending**: Use the per-K blending weight w from 01_response_matching
(either oracle or CV-selected, whichever is adopted).

**Metric**: Per-drug Pearson r, macro-averaged across drugs, evaluated on
non-anchor cells only.

## Validation checks

- Random baseline per-drug r at each K must match 01_response_matching
  (within 0.01, accounting for draw variance).
- Cell selection must be computed from training data only (no leakage of test
  drug IC50 into cell choice). Verify by checking that the selected cell set
  is identical across all test drugs within a fold.
- At K=1, all strategies except Random are deterministic within a fold; verify
  zero variance across draws.
- MaxVar cells should have measurably higher IC50 variance than random cells
  (log this as a sanity check).
- Diverse cells should have larger mean pairwise distance in PCA space than
  random cells (log this as a sanity check).

## Output

**`report/data/results.json`** schema:
```json
{
  "overall": {
    "random_baseline_r_by_k": {"1": 0.64, "3": 0.65, "5": 0.66, "10": 0.68, "20": 0.70},
    "measurement_ceiling": 0.754
  },
  "strategy_by_k": [
    {
      "strategy": "MaxVar",
      "k": 10,
      "mean_r": 0.71,
      "std_r": 0.02,
      "delta_vs_random": 0.03,
      "n_drugs": 233
    }
  ],
  "per_drug": [
    {
      "drug": "Trametinib",
      "drug_id": 1372,
      "strategy": "MaxVar",
      "k": 10,
      "mean_r": 0.75,
      "n_folds": 10
    }
  ]
}
```

**`report/data/strategy_comparison.csv`**: flat table (strategy, k, mean_r, std_r, delta_vs_random).

**`report/data/per_drug_by_strategy.csv`**: flat table (drug, drug_id, strategy, k, mean_r).

## Dependencies

- Data: `data/processed/drug_response.parquet`, omics parquets
- Splits: `external/PASO/data/10_fold_data/drug_blind/`
- Code: `src/evaluation/per_drug.py`, `src/data/splits.py`, `src/data/omics_utils.py`
- Upstream: `experiments/05_solutions/03_few_shot/01_response_matching/` (blending
  weights, random baseline for validation)

## Resources

CPU only, <1h, --mem=32G.

## How to run

```bash
~/.local/bin/uv run python3 experiments/05_solutions/03_few_shot/02_active_selection/jobs/run.py
```

## Downstream use

Identifies the best cell selection strategy for practical few-shot deployment.
If a targeted strategy reliably outperforms random at low K, this directly
reduces the experimental cost of characterizing new drugs. Feeds into:
- `05_combinations/01_moa_x_kshot` (combining within-MoA training with
  actively-selected K-shot observations)

Note: this experiment was previously planned as `06_active_learning`, now
folded into `05_solutions/03_few_shot` as a natural companion to response
matching.
