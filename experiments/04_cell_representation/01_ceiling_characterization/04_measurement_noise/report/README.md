# 04 — Measurement noise ceiling from GDSC2 replicates

## Research question

What is the within-assay measurement ceiling for per-drug Pearson r? If the same
drug is screened twice on the same cell lines, how well do the two measurements
agree? This sets the fundamental upper bound: no model can exceed the
reproducibility of the data it is trained on.

## Background

Per-drug r is bounded above by measurement reproducibility: if the assay is
noisy, even a perfect predictor cannot exceed replicate concordance. The Ridge
per-drug r (~0.645) could be near this fundamental limit — leaving little room
for improvement regardless of model class or features — or it could be far
below it, meaning algorithmic advances could close a large gap. Establishing
the measurement ceiling is essential for interpreting the representation sweep
results: if Ridge already captures >80% of the ceiling, the remaining headroom
is structurally limited, not a feature-selection problem.

## Experimental design

GDSC2 contains 9 drugs that were screened in both GDSC1 and GDSC2
on overlapping cell lines, totalling 6288 replicated drug–cell pairs.
For each drug, we compute the Pearson r between GDSC1 and GDSC2 IC₅₀ values
across shared cell lines — this is the per-drug replicate concordance.

## Results

### Overall

| Metric | Value |
|--------|:-----:|
| Per-drug replicate r (mean) | 0.7538 |
| Per-drug replicate r (std) | 0.1242 |
| Per-drug replicate r (median) | 0.7885 |
| Overall replicate r (all pairs) | 0.8798 |
| Mean absolute IC₅₀ difference | 1.39 |
| Ridge as fraction of ceiling | 83.7% |

### Per-drug breakdown

| Drug | Replicate r | N cells |
|------|:----------:|:-------:|
| Acetalax | 0.9273 | 717 |
| Oxaliplatin | 0.8825 | 717 |
| Selumetinib | 0.8372 | 699 |
| Docetaxel | 0.8296 | 669 |
| Uprosertib | 0.7885 | 668 |
| Dactinomycin | 0.7252 | 699 |
| Fulvestrant | 0.6533 | 715 |
| Ulixertinib | 0.5743 | 735 |
| GSK343 | 0.5666 | 669 |

## Interpretation

The mean per-drug replicate concordance is **r = 0.754**
(median 0.788), establishing the measurement
ceiling for within-drug cell ranking. This number means that if you screen the
same drug twice on the same cells, the two rankings agree at r ≈ 0.75 — any
prediction model achieving this level has matched the intrinsic reproducibility
of the assay.

The range across drugs is substantial,
reflecting that some drugs (Acetalax, Oxaliplatin) have highly reproducible
dose–response curves while others (GSK343, Ulixertinib) show more variability —
likely due to steep dose–response slopes near the IC₅₀ inflection point.

Our Ridge baseline achieves 83.7% of this
ceiling, confirming the paper's claim that the zero-shot model is already close
to the measurement limit. The remaining gap (16.3%)
represents the maximum room for improvement from better models before
measurement noise becomes the binding constraint.

Ridge captures 75–90% of the noise ceiling — modest room for improvement.


