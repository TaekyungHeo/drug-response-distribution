# Metric Decomposition

Establishes that global Pearson r is not a reliable measure of drug response prediction
ability, selects a primary metric to replace it, then benchmarks models against that
metric.

| # | Experiment | Status | Key result |
|---|-----------|--------|------------|
| 01 | `01_global_vs_perdrug` | Complete | Global r rejected: 68.1% between-drug variance; empirical ceiling 0.837; gap persists across all model capacities and reverses across splits |
| 02 | `02_metric_selection` | Complete | Per-drug Pearson r selected: narrowest CI (median 0.090), highest inter-metric agreement with Spearman (ρ=0.972) and Kendall τ (ρ=0.974); R² rejected (CI=0.375, poor correlation) |
| 03 | `03_baselines` | Complete | Cell-mean prior holdout oracle r=0.652 (drug-blind, test-set means); Ridge r=0.645 (drug-blind CV) ≈ practical cell-mean prior — Ridge learns mean sensitivity, not drug-specific signal |
