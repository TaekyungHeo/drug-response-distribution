# 04 — Split Robustness

Tests whether the null from `02_representation_ablation` depends on random drug-blind splits
placing structurally similar drugs across train/test boundaries.

Bemis-Murcko scaffold-stratified 5-fold CV: all drugs sharing a scaffold are held out together.

| Condition | Per-drug r | Δ |
|-----------|-----------|---|
| `no_drug` (scaffold-blind) | 0.640 ± 0.011 | — |
| `morgan_fp` (scaffold-blind) | 0.641 ± 0.011 | +0.0006 |

**Null holds under maximum structural novelty in test drugs.** Even when test drugs share
no scaffold with any training drug, Morgan fingerprints provide no benefit to within-drug
cell ranking. 219 Bemis-Murcko scaffolds; scaffold_leak_assertion passed.

See [PLAN.md](PLAN.md) for full design and run instructions.
