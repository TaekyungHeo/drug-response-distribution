# Training distribution determines the ceiling of drug-blind cancer sensitivity prediction

[![arXiv](https://img.shields.io/badge/arXiv-2605.20885-b31b1b.svg)](https://arxiv.org/abs/2605.20885)

Precision oncology requires predicting which drugs will suppress a specific tumor from its molecular profile, but drug-blind sensitivity prediction has plateaued despite increasingly complex drug representations. Here we show that this stagnation reflects a metric artifact rather than a representational bottleneck. The standard benchmark, global Pearson *r*, is dominated by between-drug potency differences that a trivial drug-mean predictor captures without any cell-specific learning. Per-drug Pearson *r*, which isolates within-drug cell ranking, reveals that no drug encoding improves over cell-only features across four independent datasets. A controlled experiment channeling mechanism-of-action identity as either a drug feature or a training-distribution constraint identifies the cause. Supplying MoA as a feature yields negligible benefit, whereas using it to stratify training raises per-drug *r* substantially for targeted kinase inhibitors, because pan-cancer co-training suppresses pathway-specific sensitivity signals. Mechanism-stratified training and response matching from pilot observations provide two deployable strategies that together recover the principal sources of predictive gain in drug-blind sensitivity prediction.

---

## Repository structure

```
├── src/                    # Shared library — imported by all experiments
│   ├── data/               # Dataset loaders, drug feature encoders
│   ├── models/             # TransformerEncoder, TransformerEncoder-GNN, MLP, multi-task variants
│   ├── training/           # Training loops and configs
│   ├── evaluation/         # per_drug_r() and other metrics
│   └── utils/              # Ridge helpers, response matching, fold loading
│
├── experiments/            # One self-contained directory per experiment
│   ├── 00_data_preparation/    # Download and preprocess raw data
│   ├── 01_metric_decomposition/# Establish per-drug r; characterize baselines
│   ├── 02_reproductions/       # PASO, DeepCDR, DrugCell bias audits
│   ├── 03_drug_feature_null/   # Drug representation ablations
│   ├── 04_cell_representation/ # Cell feature ceiling and alternatives
│   ├── 05_solutions/           # MoA training, K-shot matching, LINCS
│   └── 06_external_validation/ # BeatAML and CTRPv2 replication
│
├── scripts/                # Figure generation for the paper
├── paper/                  # LaTeX source
├── external/               # Patched PASO, DeepCDR, DrugCell for fair evaluation
└── data/
    ├── raw/        (git-ignored) downloaded source files
    └── processed/  (git-ignored) parquet matrices used by experiments
```

Each experiment directory is self-contained: it has its own script(s) and a `report/README.md` with the result and interpretation. Experiments follow the paper's narrative — see [`experiments/README.md`](experiments/README.md) for the full index and how they connect.

---

## Citation

```bibtex
@article{heo2026training,
  title   = {Training distribution determines the ceiling of drug-blind cancer sensitivity prediction},
  author  = {Heo, Taekyung},
  journal = {arXiv preprint arXiv:2605.20885},
  year    = {2026}
}
```

Heo, T. Training distribution determines the ceiling of drug-blind cancer sensitivity prediction. *arXiv* 2605.20885 (2026).

---

## License

This project is licensed under the [GNU Affero General Public License v3 (AGPL v3)](LICENSE) for non-commercial and academic use.

For commercial licensing, contact **taekyung.cs@gmail.com**. See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.
