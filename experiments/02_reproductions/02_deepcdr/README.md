# DeepCDR

Code inspection audit of DeepCDR's reported global Pearson r = 0.9211
(Liu et al., Bioinformatics 2020 / ECCB).

| # | Experiment | Purpose |
|---|-----------|---------|
| 01 | `01_snooping_audit` | Establish test-set snooping mechanism from static code analysis |

The snooping mechanism is fully established from `external/DeepCDR/prog/run_DeepCDR.py`.
Runtime reproduction is not performed: the codebase requires Python 2 syntax and
Keras 1.x. Code inspection alone is sufficient to confirm the same protocol artifact
found in PASO and DrugCell.
