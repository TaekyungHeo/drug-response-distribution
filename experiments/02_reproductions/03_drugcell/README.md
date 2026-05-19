# DrugCell

Code inspection audit of DrugCell's test-set snooping mechanism
(Kuenzi et al., Cancer Cell 2020).

| # | Experiment | Purpose |
|---|-----------|---------|
| 01 | `01_snooping_audit` | Establish snooping mechanism from static code analysis |

The snooping mechanism is fully confirmed from reading `train_drugcell.py` — no
runtime experiment is required. DrugCell's native AUC data (GDSC + CTRPv2) is not
in this repository; the code inspection alone is sufficient to establish the finding.
Metric inflation on AUC data would require a separate oracle measurement on the
native dataset, which is out of scope.
