# 04 — Multi-Omics Cell Representation

Tests whether adding CNV, metabolomics, or RPPA to RNA+mutations breaks the per-drug r
ceiling. All experiments use Ridge(α=1.0), PASO 10-fold drug-blind CV, no drug features,
per-drug Pearson r.

**Key result:** All multi-omics combinations converge to r≈0.645; max improvement is
Δ=+0.004 (RNA+mut+metabolomics). Additional modalities add no meaningful signal.

## Results

| Condition | Modalities | Features | Per-drug r | Δ vs RNA+mut |
|-----------|-----------|:--------:|:----------:|:------------:|
| RNA+mut | rna, mutations | 750 | 0.6453 | — |
| RNA+mut+CNV | rna, mutations, cnv | 750+CNV | 0.6453 | 0.000 |
| RNA+mut+metabolomics | rna, mutations, metabolomics | 750+metab | 0.6495 | +0.004 |
| RNA+mut+all | rna, mutations, cnv, metabolomics | 750+all | 0.6495 | +0.004 |

CNV adds zero signal on top of RNA+mutations. Metabolomics adds Δ=+0.004, which is
within fold-to-fold noise (fold std ≈ 0.025). No multi-omics combination crosses the
Δ>0.010 relevance threshold.

The full-dataset overlap for all modalities (RNA + mut + CNV + metabolomics) is 597
cell lines vs 687 for RNA+mut alone. The metabolomics Δ=+0.004 is partially explained
by this reduced (and potentially more homogeneous) cell set.

## Interpretation

The ceiling is not representation-limited at the modality level. Signaling state
(metabolomics) and genomic amplification (CNV) carry no information about drug response
beyond what transcriptomics already captures, at least for the Ridge model under
drug-blind CV.

## Input data

- `data/processed/rna.parquet`, `mutations.parquet`, `cnv.parquet`, `metabolomics.parquet`
- `data/processed/overlap_cell_lines.parquet` — 597-cell full-omics intersection
- `external/PASO/data/10_fold_data/drug_blind/` — PASO 10-fold drug-blind splits

## Output files

- `report/data/results_rna_mut.json`
- `report/data/results_rna_mut_cnv.json`
- `report/data/results_rna_mut_metab.json`
- `report/data/results_rna_mut_all.json`
