# Few-Shot

Source 2: direct observation of K responses for a new drug. Can K pilot IC50 values break the cell-mean prior ceiling?

| # | Experiment | Question | Key result |
|---|-----------|---------|------------|
| 01 | `01_response_matching/` | K-shot curve (K=0→50) with cell-mean blending | K=5 marginal (+0.001); reliable from K=20 (+0.025); K=50 r=0.701 |
| 02 | `02_active_selection/` | Which K cells to screen? | MaxVar at K=1: +0.021 over random; no strategy dominates across all K |
| 03 | `03_kshot_mechanism/` | K=1 in BeatAML: patient-specific vs potency signal | Cell-specific r=0.356 vs mean-potency r=0.298 (Δ=+0.059) |
