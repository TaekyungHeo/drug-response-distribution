# 06 — Objective Axis

Tests whether the training objective — not the drug representation — is the binding
constraint on within-drug cell ranking.

Morgan FP is fixed. Only the loss function changes: MSE (baseline) vs pairwise RankNet BCE.

| Condition | Per-drug r | Δ vs no_drug |
|-----------|-----------|---|
| `mlp_mse_no_drug` | 0.644 ± 0.024 | — |
| `mlp_mse_morgan` | 0.641 ± 0.025 | −0.003 |
| `mlp_ranknet_morgan` | 0.658 ± 0.023 | +0.014 |

**Switching from MSE to RankNet with the same Morgan FP features crosses the Δ=0.01 gate
(Δ=+0.014 vs no_drug, Δ=+0.017 vs mse_morgan).**

This is distinct from `02_representation_ablation`'s finding: drug representations are
irrelevant under MSE, but the ranking objective can extract a signal from the same features.
The binding constraint is the objective, not the representation itself.

See [PLAN.md](PLAN.md) for full design and run instructions.
