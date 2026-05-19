# Methodological Robustness

Is the drug-blind per-drug r=0.645 ceiling an artifact of the Ridge+MSE methodology,
or does it hold across model classes, drug-split definitions, and training objectives?

All experiments use RNA PCA(550) + mut PCA(200), PASO drug set (233 drugs),
per-drug Pearson r as the primary metric.

| # | Experiment | Question | Key result |
|---|-----------|---------|------------|
| 01 | `01_nonlinear_models/` | Does XGBoost or MLP break the Ridge ceiling? | XGBoost r=0.645 (Δ≈0), MLP r=0.639 (Δ=-0.006) — no improvement |
| 02 | `02_chemical_split/` | Does Tanimoto drug-blind split lower r? | Tanimoto r=0.660 (Δ=+0.015 vs random 0.645) — negligible chemical leakage |
| 03 | `03_ranking_loss/` | Does within-drug ranking loss improve over MSE-trained Ridge? | Ranking loss r=0.648 (Δ=+0.003 vs MSE) — not significant |

Conclusion: the ceiling is not an artifact of Ridge, random drug splits, or MSE loss.
Nonlinear models provide no benefit; scaffold-stratified splits produce essentially the
same performance; ranking-optimized training adds no meaningful lift.
