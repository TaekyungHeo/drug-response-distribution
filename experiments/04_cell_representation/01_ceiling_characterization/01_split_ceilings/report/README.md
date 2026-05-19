# 01 — Cell-blind vs drug-blind per-drug r: which split is harder?

## Research question

Cell-blind evaluation (same drugs, unseen cells) vs drug-blind evaluation
(same cells, unseen drugs) — which yields higher per-drug r, and what does
the gap tell us about the bottleneck?

## Background

Prior experiments established per-drug r as the correct metric for within-drug
cell ranking. But the magnitude of per-drug r depends on the evaluation regime.
Drug-blind splits (held-out drugs) and cell-blind splits (held-out cells)
represent categorically different generalization tasks. Before characterizing
the per-drug r ceiling, we need to understand which regime is harder — this
determines whether the ~0.645 baseline is near-optimal or substantially below
what better cell representations could achieve.

## Experimental design

- **Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200)
- **Data**: GDSC2, 10-fold PASO drug-blind splits; matched cell-blind splits

## Results

| Split | Global r | Per-drug r (mean ± std) |
|-------|:--------:|:----------------------:|
| Drug-blind | 0.340 | 0.645 ± 0.025 |
| Cell-blind | 0.223 | 0.438 ± 0.028 |

**Drug-mean cheat predictor**: global r = 0.845, per-drug r = NaN

## Interpretation

**Counterintuitive finding**: drug-blind per-drug r (0.645)
exceeds cell-blind per-drug r (0.438).

This inverts the conventional wisdom that drug-blind is the harder evaluation.
The conventional ranking holds for global r (where drug-blind is harder because
test-drug means are unknown), but reverses for per-drug r. The reason: in
drug-blind evaluation, all training cells appear in both train and test — the
model leverages cell familiarity to rank cells for new drugs. In cell-blind
evaluation, the model must extrapolate to cell-state regions absent from
training, which is harder for within-drug ranking.

This confirms that within-drug cell ranking depends primarily on the cell-state
representation, not on drug identity. Cell-side improvement is the wrong
intervention for drug-blind per-drug r — the model already sees all cells.


