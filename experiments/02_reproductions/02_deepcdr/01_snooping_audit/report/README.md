# DeepCDR test-set snooping audit

## Research question

Does DeepCDR (Liu et al., Bioinformatics 2020) use the test set for model selection,
and if so, does this inflate reported performance?

## Background

Code inspection of DeepCDR revealed that `run_DeepCDR.py:278` passes test data
directly as the Keras `validation_data`. The custom `MyCallback` computes Pearson r on
these test predictions each epoch and triggers early stopping against test performance.
No separate validation set exists in the codebase. DeepCDR reported mixed-set global
Pearson r = 0.9211.

This experiment quantifies the impact using a Ridge proxy on GDSC2, isolating the
protocol artifact independently of the original architecture.

## Experimental design

Ridge regression (RNA PCA(550) + mutation PCA(200)) was evaluated on GDSC2 under
drug-blind cross-validation. Three conditions were compared. The **oracle** uses the
per-drug mean predictor, establishing the upper bound for global r without any
cell-level learning. The **fair protocol** uses a held-out validation set for model
selection, with the test set evaluated once. The **snooping protocol** selects the
model based on test-set performance, reproducing DeepCDR's evaluation structure.
The gap between fair and snooping protocols measures the inflation attributable to
test-set selection.

## Results

| Protocol | Global r | Per-drug r |
|----------|:--------:|:----------:|
| Oracle (drug-mean) | 0.8265 | — |
| Fair (validation selection) | 0.8522 | 0.6379 |
| Snooping (test selection) | 0.8519 | — |
| Snooping delta | -0.0004 | — |

In the Ridge proxy, test-set snooping contributes a delta of -0.0004
to global r, negligible for a linear model because Ridge has no epoch-level overfitting
dynamics. The snooping inflation is architecture-dependent: deep models with hundreds of
training epochs and stochastic optimization are more susceptible than Ridge.

The critical finding is the protocol pattern rather than the magnitude in our proxy.
DeepCDR's codebase structurally conflates validation and test data, meaning the reported
r = 0.9211 was selected against test performance. Combined
with identical patterns found in PASO and DrugCell, this establishes test-set snooping
as a systemic issue in the drug response prediction field.

The fair per-drug r of 0.6379 from our
proxy is consistent with the all-drug Ridge baseline (~0.645), confirming that
DeepCDR's architecture does not fundamentally change within-drug ranking performance.

