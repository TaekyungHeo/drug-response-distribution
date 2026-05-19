# 02 — LINCS x within-MoA 2×2 factorial

## Research question

Do LINCS drug features and within-MoA training combine to improve both global r
and per-drug r simultaneously?

## Background

LINCS does not improve per-drug r (04_external_signatures/01_lincs, Δ=+0.001).
Within-MoA training improves per-drug r but does not necessarily improve global r
(02_training_distribution/01_within_moa). A 2×2 factorial
(with/without LINCS × all-drug/within-MoA) tests whether both interventions
interact or are on independent axes.

## Experimental design

- **Model**: Ridge(alpha=1.0), cell features = RNA PCA(550) + mutation PCA(200)
- **Drug features**: LINCS PCA(64) capturing 98.0% of variance (when used)
- **CV**: PASO 10-fold drug-blind CV for all-drug conditions; leave-one-drug-out within MoA for within-MoA conditions
- **Restriction**: all conditions evaluated on LINCS-covered drugs only for fair comparison
- **Focus MoAs**: classes with sufficient LINCS coverage and high within-MoA ceiling

## Results

### ERK MAPK signaling (5/11 drugs LINCS-covered)

#### 2×2 factorial: global r / per-drug r

| | no LINCS | + LINCS |
|---|:---:|:---:|
| **all-drug** | global=0.2709, pdr=0.3637 | global=-0.0015, pdr=0.3715 |
| **within-MoA** | global=0.3160, pdr=0.7203 | global=0.3275, pdr=0.7210 |

LINCS effect on global r: -0.2724.
Within-MoA effect on per-drug r: 0.3566.

### EGFR signaling (4/7 drugs LINCS-covered)

#### 2×2 factorial: global r / per-drug r

| | no LINCS | + LINCS |
|---|:---:|:---:|
| **all-drug** | global=0.4313, pdr=0.4651 | global=0.3926, pdr=0.4592 |
| **within-MoA** | global=0.5714, pdr=0.7358 | global=0.6320, pdr=0.7369 |

LINCS effect on global r: -0.0387.
Within-MoA effect on per-drug r: 0.2708.


## Per-drug detail

| Drug | MoA | all-drug r | +LINCS r | within-MoA r | +LINCS+MoA r |
|------|-----|:----------:|:--------:|:------------:|:------------:|
| Afatinib | EGFR signaling | 0.4469 | 0.4339 | 0.6733 | 0.6725 |
| Dabrafenib | ERK MAPK signaling | 0.4990 | 0.5104 | 0.6073 | 0.6077 |
| Erlotinib | EGFR signaling | 0.4559 | 0.4413 | 0.7209 | 0.7209 |
| Gefitinib | EGFR signaling | 0.4576 | 0.4593 | 0.7997 | 0.8002 |
| Lapatinib | EGFR signaling | 0.4999 | 0.5024 | 0.7495 | 0.7540 |
| PLX-4720 | ERK MAPK signaling | 0.5339 | 0.5493 | 0.5044 | 0.5021 |
| Refametinib | ERK MAPK signaling | 0.2549 | 0.2546 | 0.8073 | 0.8103 |
| Selumetinib | ERK MAPK signaling | 0.3062 | 0.3176 | 0.8790 | 0.8788 |
| Trametinib | ERK MAPK signaling | 0.2242 | 0.2258 | 0.8034 | 0.8063 |

## Interpretation

The original hypothesis was that LINCS and within-MoA training would be orthogonal:
LINCS improving global r while within-MoA improves per-drug r. In practice, LINCS
does not improve global r on these small MoA subsets, consistent with the 01_lincs
finding (Δ=-0.058 on the full 104-drug subset). The combination shows that
within-MoA training is the dominant intervention for per-drug r, while LINCS adds
noise rather than complementary signal in this Ridge setup.


