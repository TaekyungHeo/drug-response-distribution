# PASO: Drug Response Prediction Model
![PASO](Figs/Fig1/poster.jpg)

## Overview
<p style="text-align: justify;">
Individualized prediction of cancer drug sensitivity is of vital importance in precision medicine. While numerous predictive methodologies for cancer drug response have been proposed, the precise prediction of an individual patient's response to drug and a thorough understanding of differences in drug responses among individuals continue to pose significant challenges. This study introduced a deep learning model PASO, which integrated transformer encoder, multi-scale convolutional networks and attention mechanisms to predict the sensitivity of cell lines to anticancer drugs, based on the omics data of cell lines and the SMILES representations of drug molecules. First, we use statistical methods to compute the differences in gene expression, gene mutation, and gene copy number variations between within and outside biological pathways, and utilized these pathway difference values as cell line features, combined with the drugs' SMILES chemical structure information as inputs to the model. Then the model integrates various deep learning technologies multi-scale convolutional networks and transformer encoder to extract the properties of drug molecules from different perspectives, while an attention network is devoted to learning complex interactions between the omics features of cell lines and the aforementioned properties of drug molecules. Finally, a multilayer perceptron (MLP) outputs the final predictions of drug response. Our model exhibits higher accuracy in predicting the sensitivity to anticancer drugs comparing with other methods proposed recently. It is found that PARP inhibitors, and Topoisomerase I inhibitors were particularly sensitive to SCLC when analyzing the drug response predictions for lung cancer cell lines. Additionally, the model is capable of highlighting biological pathways related to cancer and accurately capturing critical parts of the drug's chemical structure. We also validated the model's clinical utility using clinical data from The Cancer Genome Atlas. In summary, the PASO model suggests potential as a robust support in individualized cancer treatment.
</p>

## Project Structure

```tree
.
├── data/                   # Data directory for model training and evaluation
│
├── data_preprocessing/     # Data preprocessing scripts and notebooks
│                           # (GEP, CNV, Mutation, and TCGA data)
│
├── Figs/                  # Figures and visualization results (Fig 1-8)
│
├── models/                # Implementation of all models and their results
│   │
│   ├── LightGBM & XGBoost/   # Advanced machine learning methods
│   │
│   ├── PASO/             # Our proposed PASO model implementations
│   │
│   ├── PathDSP/           # Deep learning baseline models
│   │
│   ├── Precily/           # Deep learning baseline models
│   │
│   └── RF & SVM/       # Classical machine learning methods
│
└── utils/              # Utility functions and scripts
```
## Directory Details

### Data Directory (`data/`)
Contains all the necessary data for training and evaluating the models. The data includes:
- Gene expression profiles
- Copy number variation data
- Mutation data
- TCGA dataset
- Model hyperparameters (optimal parameters for PASO and baselines)

### Data Preprocessing (`data_preprocessing/`)
Contains scripts and documentation for preprocessing different types of data:
- GEP processing: Scripts for preprocessing gene expression profile data
- CNV processing: Tools for handling copy number variation data
- Mutation data processing: Scripts for mutation data preparation
- TCGA processing: Scripts for TCGA data preprocessing

### Figures (`Figs/`)
Stores all figures and visualizations, including:
- Main manuscript figures (Figure 1-8) with final versions used in the paper
- Supplementary Information figures and additional visualizations
- Performance comparison plots between models
- Detailed data analysis results and visualizations

### Models (`models/`)
Contains implementations and results for all models used in the study:

1. PASO Models:
   - Multiple versions of our proposed PASO architecture
   - Training scripts and configurations
   - Results and model checkpoints

2. Machine Learning Baselines:
   - Random Forest
   - Support Vector Machine (SVM)
   - LightGBM
   - XGBoost
   - Performance results and comparisons

3. Deep Learning Baselines:
   - Precily implementation
   - PathDSP implementation
   - Comparative results and analysis

## Requirements
- Python 3.10.13
- PyTorch 2.1.0
- pytoda 1.1.3
- pandas 2.1.2
- numpy 1.26.0
- jupyter 1.0.0
- scikit-learn 1.3.2
- scipy 1.11.3 
- seaborn 0.12.2  
- xgboost 2.1.2
- lightgbm 4.5.0

## Quick Start
```bash
# Install requirements
pip install -r requirements.txt

# Run training
python models/PASO/train/train_PASO_GEP.py

# Run prediction
python models/PASO/test/PASO_predict & save_attention_result.py

# other baselines can be run in a similar way

```