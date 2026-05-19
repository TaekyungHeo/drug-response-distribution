# PASO

Artifact audit of PASO's reported drug-blind r = 0.745 (Wu et al., PLoS Comput Biol 2025).

| # | Experiment | Purpose |
|---|-----------|---------|
| 01 | `01_reproduction` | Run PASO's original code on their data to confirm r ≈ 0.745 |
| 02 | `02_decomposition` | Decompose 0.745 into snooping inflation and best-fold cherry-picking |

`01_reproduction` establishes that PASO's code is reproducible from their published
splits before attributing the headline number to protocol artifacts.
`02_decomposition` is the primary contribution: it quantifies how much of r = 0.745
is accounted for by snooping and single-fold selection rather than model quality.
