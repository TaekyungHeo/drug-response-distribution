# PLAN: DeepCDR r = 0.9211 — Test-Set Snooping Code Inspection

## Core finding

Liu et al. (Bioinformatics 2020, ECCB) report an overall Pearson r of **0.9211** for
DeepCDR on GDSC (GDSC1). The reported metric is inflated by two independent artifacts:

1. **Test-set snooping**: DeepCDR selects its final checkpoint by monitoring test-set
   Pearson r at every epoch across 500 epochs. No validation set is held out. The
   reported r = 0.9211 is the best-ever test-set Pearson, not a generalization estimate.

2. **Global-r metric inflation**: GDSC IC₅₀ variance is dominated by between-drug
   differences (~68%). A drug-mean oracle achieves global r ≈ 0.83 with zero
   cell-sensitivity ability. Any model evaluated by global r competes primarily against
   this trivial baseline, not the clinically relevant task of ranking cell lines within
   a drug. DeepCDR reports global r and does not report per-drug r.

Runtime reproduction is not performed. The codebase requires Python 2 syntax and
Keras 1.x, making runtime replication impractical. Code inspection alone fully
establishes the snooping mechanism.

## Code inspection: the snooping mechanism

File: `external/DeepCDR/prog/run_DeepCDR.py`

```python
# Line 278: test data passed directly as Keras validation_data
validation_data = [[X_drug_feat_data_test, X_drug_adj_data_test,
                    X_mutation_data_test, X_gexpr_data_test,
                    X_methylation_data_test], Y_test]

# Line 281: training runs with test data as validation
model = ModelTraining(model, ..., validation_data, nb_epoch=500)
```

Inside `MyCallback.on_epoch_end` (lines 223–228):
```python
y_pred_val = self.model.predict(self.x_val)          # x_val IS x_test
pcc_val = pearsonr(self.y_val, y_pred_val[:,0])[0]   # y_val IS y_test
if pcc_val > self.best:
    self.best = pcc_val
    self.best_weight = self.model.get_weights()       # saved by test r
```

`on_train_end` (line 213): restores the checkpoint with the highest test-set Pearson
as the final model. `ModelEvaluate` then evaluates on the same `Y_test`. The
reported r = 0.9211 is the best-ever test-set Pearson across 500 epochs, not a
generalization estimate.

## Data

DeepCDR uses GDSC1 (2019 vintage): ~223 drugs, 561 cell lines, ~73K (cell, drug) pairs.
Single 95/5 train/test partition stratified by cancer type — no cross-validation, no
drug-blind constraint. This is a mixed-set evaluation: test drugs all appear in training.

## Relationship to PASO and DrugCell

All three models use the same mechanism: monitor test-set Pearson across epochs, save
the best checkpoint, report that checkpoint's test-set metric as the final result.
PASO provides a quantitative decomposition. DeepCDR and DrugCell confirm the pattern
independently, establishing it as systemic across institutions and publication venues
(Bioinformatics, Cancer Cell, PLoS Computational Biology).

## What this experiment produces

A documented code audit with file and line-level citations. No runtime outputs.
