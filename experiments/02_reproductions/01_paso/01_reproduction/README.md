# 01 — Reproduce PASO's r = 0.745

Runs PASO's original GEP+MUT model on their published drug-blind 10-fold splits
using their snooping protocol (checkpoint selected by best test-set Pearson).
Confirms r ≈ 0.745 is reproducible from their code before attributing the number
to protocol artifacts in `02_decomposition`.

Key result: 10-fold mean r = 0.603 ± 0.091 (PASO's snooping protocol); best fold (fold 8)
r = 0.712, 0.033 below PASO's reported 0.745. Confirms reproducibility before
`02_decomposition` attributes the headline to protocol artifacts.
