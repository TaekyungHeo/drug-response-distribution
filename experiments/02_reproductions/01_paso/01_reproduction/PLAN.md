# PLAN: Reproduce PASO's r = 0.745

## What this experiment does

Runs PASO's original GEP+MUT model (`PASO_GEP_MUT`) on their published drug-blind
10-fold splits to confirm r ≈ 0.745 is reproducible from their code.
Uses PASO's snooping protocol: checkpoint selected by best test-set Pearson each epoch.

The snooping mechanism (`run.py`, lines 125–133):
```python
if r > max_pearson:
    max_pearson = r          # track best TEST Pearson across all epochs
    best_epoch = epoch
    torch.save(model.state_dict(), ".../best_pearson.pt")
```
No validation set is held out. The reported r per fold is the maximum test-set Pearson
observed across all 200 epochs.

## Prerequisites

### PASO submodule

```bash
cd ~/multi-onco
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init external/PASO
```

Expected files:
```
external/PASO/data/10_fold_data/drug_blind/DrugBlind_train_Fold{0-9}.csv
external/PASO/data/10_fold_data/drug_blind/DrugBlind_test_Fold{0-9}.csv
external/PASO/data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv
external/PASO/data/MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv
external/PASO/data/ccle-gdsc.smi
external/PASO/data/MUDICUS_Omic_619_pathways.pkl
external/PASO/data/smiles_language/tokenizer_customized/
external/PASO/data/best_hyp/proposed_model/PASO_GEP_MUT.json
```

### PASO dependencies

`pytoda==1.1.3` is PASO's custom SMILES tokenizer. Test before submitting:
```bash
~/.local/bin/uv run --with "pytoda==1.1.3" python3 -c "import pytoda; print('ok')"
```

If the build fails (pytoda requires Cython), install into a venv:
```bash
python3 -m venv external/PASO/.venv
external/PASO/.venv/bin/pip install pytoda==1.1.3
external/PASO/.venv/bin/python3 experiments/02_reproductions/01_paso/01_reproduction/jobs/run.py
```

## Hyperparameters (from PASO_GEP_MUT.json)

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 512 (overrides JSON's 192) |
| LR | 0.001 |
| Optimizer | Adam |
| Folds | 10 |
| GEP standardize | true |
| MUT standardize | false |
| SMILES padding | 256 |
| Dropout | 0.3 |
| Hidden sizes | [1024, 512, 128] |

## How to run

spark1 and spark2 do not share disks. Set up the submodule on both nodes before submitting.

```bash
cd ~/multi-onco
sbatch experiments/02_reproductions/01_paso/01_reproduction/jobs/sbatch.sh
```

Runtime: ~15–20 h on a single GPU (10 folds × 200 epochs).

## Output

Per-fold results written to:
```
external/PASO/models/PASO/result/reproduce_DrugBlind_GEP_MUT/Fold{N}/results/pearson.json
```

Aggregated summary:
```
experiments/02_reproductions/01_paso/01_reproduction/report/data/results.json
```

```json
{
  "fold_best_test_r": [<fold0_r>, ..., <fold9_r>],
  "mean": <float>,
  "std": <float>,
  "best_fold_r": <float>,
  "best_fold_index": <int, 1-indexed>
}
```

## Expected results

| Metric | Expected |
|--------|----------|
| Best single fold r | ≈ 0.745 (PASO paper headline) |
| 10-fold mean | ≈ 0.55–0.60 (not reported in paper) |
