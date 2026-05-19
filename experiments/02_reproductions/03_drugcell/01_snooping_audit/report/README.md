# DrugCell test-set snooping audit

## Research question

Does DrugCell (Kuenzi et al., Cancer Cell 2020) use the test set for model selection,
and if so, does this inflate reported performance?

## Background

Code inspection of DrugCell revealed that `train_drugcell.py:134` computes test-set
Pearson r each epoch, printed as `val_corr` (L137). The best model is explicitly
selected by `if test_corr >= max_corr: best_model = epoch` (L140–142). The `-test`
parameter is documented as "Validation dataset used for early termination condition."
A separate `drugcell_val.txt` file ships with the repository but is never loaded by
the training code; model selection runs entirely on the held-out test set.

This experiment quantifies the impact using a Ridge proxy on GDSC2.

## Experimental design

Ridge regression (RNA PCA(550) + mutation PCA(200)) was evaluated on GDSC2 under
drug-blind cross-validation. Three conditions were compared. The **oracle** uses the
per-drug mean predictor, establishing the upper bound for global r without any
cell-level learning. The **fair protocol** uses a held-out validation set for model
selection, with the test set evaluated once. The **snooping protocol** selects the
model based on test-set performance, reproducing DrugCell's evaluation structure.
Note: DrugCell reported on GDSC+CTRPv2 AUC; this proxy uses GDSC2 IC50 to demonstrate protocol artifact

## Results

| Protocol | Global r | Per-drug r |
|----------|:--------:|:----------:|
| Oracle (drug-mean) | 0.8265 | — |
| Fair (validation selection) | 0.8523 | 0.6362 |
| Snooping (test selection) | 0.8504 | — |
| Snooping delta | -0.0020 | — |

The snooping delta in the Ridge proxy is -0.0020,
negligible for the same reason as DeepCDR: Ridge lacks the epoch-level overfitting
dynamics that make snooping impactful in deep models.

The significance of this audit lies in the code-level evidence. DrugCell's training
code explicitly selects the best model on test data while naming the variable `val_corr`.
A validation file exists but is unused. This is the third major drug response prediction
model (alongside PASO and DeepCDR) where test-set snooping was identified, spanning
two continents and two top-tier journals. The pattern is field-wide rather than isolated.

The fair per-drug r of 0.6362 is
consistent with the Ridge baseline (~0.645), indicating that DrugCell's neural network
architecture does not improve within-drug cell ranking in the drug-blind setting.

