# 03 — Within-lineage training: pan-cancer vs lineage-specific models

## Research question

Does training a separate Ridge model per cancer lineage (e.g., Breast only,
Lung only) improve within-lineage per-drug r compared to the pan-cancer model?
If pan-cancer training suppresses lineage-specific signals (analogous to
MoA-stratified training suppressing pathway signals), lineage-specific models
should outperform.

## Background

The 05_solutions experiment shows that MoA-stratified training — restricting
training drugs to the same MoA class — improves per-drug r by +0.296–+0.375.
The mechanism is that pan-cancer, multi-drug training forces the model to learn
a compromise representation across diverse drug-cell relationships. A parallel
question arises for the cell axis: does pan-cancer training dilute
lineage-specific signals? If so, training one model per lineage should yield
higher within-lineage per-drug r, just as training per MoA yielded higher
per-MoA per-drug r.

## Experimental design

- **Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200)
- **Data**: GDSC2, per-lineage 5-fold cell-blind CV within each lineage
- **Comparison**: Pan-cancer 10-fold drug-blind baseline (per-drug r = 0.6453)

## Results

| Lineage | Within-lineage per-drug r | N cells (train) | N drugs |
|---------|:------------------------:|:---------------:|:-------:|
| Breast | 0.5705 | 47 | 24 |
| CNS | 0.6433 | 36 | 24 |
| Colorectal | 0.5541 | 42 | 24 |
| Hematologic | 0.5235 | 109 | 24 |
| Lung | 0.5981 | 131 | 24 |
| Skin | 0.4837 | 36 | 24 |

**Pan-cancer baseline**: 0.6453 ± 0.0246

## Interpretation

Within-lineage training generally yields **lower** per-drug r than the
pan-cancer model. This is the opposite of the MoA-stratified finding, and
the explanation is straightforward: lineage restriction reduces the number
of training cell lines (from ~960 to 30–200), and per-drug r depends on
having diverse cell states in training. Unlike MoA-stratified training
(which restricts drugs but keeps all cells), lineage restriction removes
the very cell-state diversity that drives accurate within-drug ranking.

This confirms that the pan-cancer fragility signal — which enables high
per-drug r — requires cross-lineage cell diversity. The MoA-stratified
training gain is specific to restricting the *drug* axis (same cells,
fewer drugs) rather than the *cell* axis (fewer cells, same drugs).


