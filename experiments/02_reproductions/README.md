# Reproductions

Three published drug response prediction models independently exhibit the same two
measurement artifacts: **global-r metric inflation** and **test-set snooping**.
This is not an isolated design mistake — it is a protocol pattern across the field.

## Artifacts

**Global-r metric inflation**: On GDSC IC₅₀, 68% of variance is between-drug. A
drug-mean oracle (no cell-sensitivity ability whatsoever) achieves global r ≈ 0.84.
Any model evaluated by global r is competing mostly against this trivial baseline,
not against the clinically relevant task of ranking cell lines within a drug.

**Test-set snooping**: All three models select the final checkpoint by monitoring
test-set Pearson r across training epochs. No held-out validation set is used.
The reported metric is the best test-set r ever observed, not a fair estimate of
generalization.

## Sub-groups

| Directory | Model | Reference | Approach |
|-----------|-------|-----------|----------|
| `01_paso/` | PASO (r = 0.745) | Wu et al., PLoS CB 2025 | Quantitative decomposition (global r + per-drug r) |
| `02_deepcdr/` | DeepCDR (r = 0.9211) | Liu et al., Bioinformatics 2020 | Code inspection audit |
| `03_drugcell/` | DrugCell | Kuenzi et al., Cancer Cell 2020 | Code inspection audit |

PASO provides the primary quantitative decomposition: `01_reproduction` confirms
r ≈ 0.745 is reproducible from PASO's own code, and `02_decomposition` decomposes
the headline into snooping inflation and best-fold cherry-picking on a drug-blind
split. DeepCDR and DrugCell extend the finding to two additional independent
codebases via static code analysis, establishing the pattern as systemic. Runtime
reproduction of DeepCDR is not performed (Python 2 / Keras 1.x); the snooping
mechanism is fully established from code inspection alone.
