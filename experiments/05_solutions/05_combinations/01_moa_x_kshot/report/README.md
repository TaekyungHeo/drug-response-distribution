# 01 — MoA x K-shot Factorial

## Research question

Do within-MoA training and K-shot response matching combine additively, or do
they exploit the same signal and saturate at the same ceiling?

## Background

Within-MoA training (02_training_distribution/01_within_moa) improves per-drug r
by specializing the training distribution to a single mechanism. K-shot response
matching (03_few_shot/01_response_matching) improves per-drug r by finding the
nearest training drug by response similarity across K pilot cells. These operate
through different mechanisms: training distribution restriction vs. post-hoc
prediction refinement. If they address different variance components, their
combination should be additive; if they identify the same informative training
drugs (those sharing mechanism and response profile), they will saturate at the
same ceiling.

## Experimental design

- **Model**: Ridge(alpha=1.0), RNA PCA(550) + mutation PCA(200), cell features only
- **CV**: PASO 10-fold drug-blind for all-drug conditions; within-MoA LOO for within-MoA conditions
- **Conditions**: all-drug baseline, within-MoA only, K-shot only, combined
- **Measurement ceiling (r_yy)**: 0.754
- **Additivity criterion**: combined > max(individual) by > 0.005

## Results

### EGFR signaling (n=7 drugs)

| Condition | K | per-drug r | optimal w |
|-----------|--:|:----------:|:---------:|
| All-drug baseline | — | 0.4251 | — |
| Within-MoA only | — | 0.7993 | — |
| K-shot only | 0 | 0.4248 | 0.0 |
| K-shot only | 5 | 0.4626 | 0.24 |
| K-shot only | 10 | 0.5096 | 0.38 |
| K-shot only | 20 | 0.6012 | 0.53 |
| K-shot only | 50 | 0.7594 | 0.87 |
| Combined | 0 | 0.7994 | 0.0 |
| Combined | 5 | 0.8004 | 0.39 |
| Combined | 10 | 0.7999 | 0.48 |
| Combined | 20 | 0.8011 | 0.46 |
| Combined | 50 | 0.7998 | 0.54 |

**Redundant**: combined (0.8011 at K=20)
does not meaningfully exceed max(individual) = 0.7993.

### ERK MAPK signaling (n=11 drugs)

| Condition | K | per-drug r | optimal w |
|-----------|--:|:----------:|:---------:|
| All-drug baseline | — | 0.4273 | — |
| Within-MoA only | — | 0.7233 | — |
| K-shot only | 0 | 0.4269 | 0.0 |
| K-shot only | 5 | 0.6000 | 0.37 |
| K-shot only | 10 | 0.6584 | 0.43 |
| K-shot only | 20 | 0.7328 | 0.59 |
| K-shot only | 50 | 0.8135 | 0.71 |
| Combined | 0 | 0.7229 | 0.0 |
| Combined | 5 | 0.7823 | 0.72 |
| Combined | 10 | 0.7873 | 0.79 |
| Combined | 20 | 0.7921 | 0.82 |
| Combined | 50 | 0.8039 | 0.84 |

**Redundant**: combined (0.8039 at K=50)
does not meaningfully exceed max(individual) = 0.8135.


## Per-drug detail (at max K)

| Drug | MoA | all-drug r | within-MoA r | K-shot r | combined r |
|------|-----|:----------:|:------------:|:--------:|:----------:|
| AZD3759 | EGFR signaling | 0.4618 | 0.7653 | 0.7482 | 0.7630 |
| Afatinib | EGFR signaling | 0.4407 | 0.8279 | 0.7506 | 0.8274 |
| Erlotinib | EGFR signaling | 0.4405 | 0.7379 | 0.7013 | 0.7371 |
| Gefitinib | EGFR signaling | 0.4550 | 0.8340 | 0.7561 | 0.8356 |
| Lapatinib | EGFR signaling | 0.5151 | 0.7807 | 0.7575 | 0.7865 |
| Osimertinib | EGFR signaling | 0.4264 | 0.8234 | 0.7792 | 0.8214 |
| Sapitinib | EGFR signaling | 0.2343 | 0.8264 | 0.8225 | 0.8280 |
| Dabrafenib | ERK MAPK signaling | 0.4981 | 0.6470 | 0.6401 | 0.7592 |
| KRAS (G12C) Inhibitor-12 | ERK MAPK signaling | 0.7304 | 0.2835 | 0.7398 | 0.4065 |
| PD0325901 | ERK MAPK signaling | 0.2145 | 0.8682 | 0.9375 | 0.9409 |
| PLX-4720 | ERK MAPK signaling | 0.5487 | 0.5110 | 0.6401 | 0.7087 |
| Refametinib | ERK MAPK signaling | 0.2324 | 0.8383 | 0.9238 | 0.9242 |
| SB590885 | ERK MAPK signaling | 0.6143 | 0.5650 | 0.6567 | 0.6856 |
| SCH772984 | ERK MAPK signaling | 0.3453 | 0.8813 | 0.9166 | 0.9157 |
| Selumetinib | ERK MAPK signaling | 0.3000 | 0.8603 | 0.8646 | 0.8948 |
| Trametinib | ERK MAPK signaling | 0.2127 | 0.8609 | 0.9552 | 0.9552 |
| Ulixertinib | ERK MAPK signaling | 0.4891 | 0.8167 | 0.8406 | 0.8246 |
| VX-11e | ERK MAPK signaling | 0.5104 | 0.8200 | 0.8335 | 0.8275 |

## Interpretation

If the combination exceeds max(individual), the two methods exploit different
information axes: within-MoA constrains the training distribution (mechanism),
while K-shot refines predictions within that distribution (observed potency).
If the combination equals max(individual), both methods ultimately identify the
same informative training drugs, and the simpler method suffices.

All results bounded by the measurement ceiling (0.754).


