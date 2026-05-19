# PLAN: DrugCell — Test-Set Snooping Code Inspection

## Core finding

Kuenzi et al. (Cancer Cell 2020) report Pearson and Spearman correlations for DrugCell
on GDSC + CTRPv2 combined data. The snooping mechanism is established entirely from
static analysis of `train_drugcell.py` — no runtime reproduction is required or planned.

## Code inspection: the snooping mechanism

File: `external/DrugCell/code/train_drugcell.py`

```python
# Lines 140–142: best epoch tracked by test correlation
if test_corr >= max_corr:
    max_corr = test_corr
    best_model = epoch
```

Every epoch is saved to disk (line 115):
```python
torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')
```

At training end, the best epoch is printed (line 146):
```python
print("Best performed model (epoch)\t%d" % best_model)
```

The reported evaluation corresponds to the checkpoint with the highest test-set
correlation across all training epochs.

## Why `drugcell_val.txt` does not help

The repository ships a separate `drugcell_val.txt` file. However, `train_drugcell.py`
neither loads nor uses it. No validation split is applied despite the file being present.
The snooping is a factual omission in the code, not a judgment call about design.

## Relationship to PASO and DeepCDR

All three models use the same mechanism: monitor test-set Pearson across epochs, save
the best checkpoint, report that checkpoint's test-set metric. PASO and DeepCDR provide
quantitative decompositions with oracle baselines. DrugCell extends the pattern to a
third independent codebase, confirming it is not isolated to any one architecture or
research group.

## Data note

DrugCell uses GDSC + CTRPv2 combined AUC data (509K pairs, 1,235 cell lines, 684 drugs).
This dataset is not in the repository. Metric inflation magnitude on AUC data may differ
from the IC₅₀-based figure (≈ 0.84) established on GDSC2; the between-drug variance
fraction would need to be measured separately on the native AUC data.

## What this experiment produces

A documented code audit with line-level citations. No runtime outputs.
