import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import optuna
import os
from typing import List, Tuple, Any
import logging
from pytoda.smiles import SMILESTokenizer

from utils.loss_functions import pearsonr

# %%
SEED = 9527
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIRECTORY = ''
PROJECT_NAME = 'Precily_comparison'
INPUT_DIM = 875
MAX_EPOCHS = 200
BATCH_SIZE = 512
PATIENCE = 5


# %%
class PrecilyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.FloatTensor(targets.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class PrecilyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float):
        super(PrecilyModel, self).__init__()

        layers = []
        layers.extend([
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])

        current_dim = 512
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# %%

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    model.train()
    total_loss = 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        targets = targets.view(-1, 1)  # Reshape targets to match outputs
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float | Any, float | Any, Any, Tensor, Tensor]:
    model.eval()

    total_loss = 0
    with torch.no_grad():
        labels = []
        preds = []
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            targets = targets.view(-1, 1)  # Reshape targets to match outputs
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            labels.append(targets)
            preds.append(outputs)
        preds_total = torch.cat([p.cpu() for preds in preds for p in preds])
        labels_total = torch.cat([l.cpu() for label in labels for l in label])

        pcc = pearsonr(preds_total, labels_total)
        r2 = r2_score(preds_total, labels_total)
        mse = mean_squared_error(preds_total, labels_total)

    return mse, r2, pcc, preds_total, labels_total
#
# def validate_(model: nn.Module,
#              val_loader: DataLoader,
#              criterion: nn.Module,
#              device: torch.device) -> float:
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for features, targets in val_loader:
#             features, targets = features.to(device), targets.to(device)
#             outputs = model(features)
#             targets = targets.view(-1, 1)  # Reshape targets to match outputs
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#             pcc = pearsonr(outputs, targets)
#             r2 = r2_score(outputs, targets)
#             mse = mean_squared_error(outputs, targets)
#
#     return total_loss / len(val_loader)

# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

        self.best_mse = float('inf')
        self.best_r2 = float('-inf')
        self.best_pcc = float('-inf')
        self.best_preds = None

    def __call__(self, mse, r2, pcc, preds):
        if self.best_loss is None:
            self.best_loss = mse
            self.best_mse = mse
            self.best_r2 = r2
            self.best_pcc = pcc
            self.best_preds = preds.cpu().numpy()
        elif mse > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = mse
            self.counter = 0

        if mse < self.best_mse:
            self.best_mse = mse
            self.best_preds = preds.cpu().numpy()
        if r2 > self.best_r2:
            self.best_r2 = r2
        if pcc > self.best_pcc:
            self.best_pcc = pcc

        return self.early_stop


# %%

def create_objective(X_train: pd.DataFrame,
                     Y_train: pd.DataFrame,
                     X_val: pd.DataFrame,
                     Y_val: pd.DataFrame,
                     device: torch.device):
    def objective(trial):
        # Define hyperparameter search space
        n_layers = trial.suggest_int('n_layers', 2, 6)
        # Use suggest_float with step for discrete uniform parameter
        dropout_rate = trial.suggest_float('drop_rate', 0.0, 0.5, step=0.1)
        # Use suggest_float with log=True for loguniform parameter
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        hidden_layers = [
            trial.suggest_int(f'units_{i}', 128, 256, step=4)
            for i in range(n_layers)
        ]

        train_dataset = PrecilyDataset(X_train, Y_train)
        val_dataset = PrecilyDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = PrecilyModel(INPUT_DIM, hidden_layers, dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=PATIENCE)

        best_val_loss = float('inf')

        for epoch in range(MAX_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _, _, _ = validate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # early stopping
            # if early_stopping(val_loss):
            #     break

            # Optuna剪枝
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_loss

    return objective


# %%
# Load data
df_smiles_origin = pd.read_csv('../../data/DrugName_AND_Canonical_SMILES.csv')
df_GEP = pd.read_csv("../../data/GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv")
# CNV
# df_CNV = pd.read_csv('../data/CNV_Cardinality_analysis_of_variance_Latest_MEDICUS.csv')
# MUT
# df_MUT = pd.read_csv('../data/MUT_cardinality_analysis_of_variance_Only_MEDICUS.csv')

# Load SMILES language
smiles_language_filepath = '../../data/smiles_language/tokenizer_customized'
smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
smiles_language.set_encoding_transforms(
    add_start_and_stop=True,
    padding=True,
    padding_length=256,
)
smiles_language.set_smiles_transforms(
    augment=False,
    canonical=False,
    kekulize=False,
    all_bonds_explicit=False,
    all_hs_explicit=False,
    remove_bonddir=False,
    remove_chirality=False,
    selfies=False,
    sanitize=False,
)
smiles_language.add_dataset(df_smiles_origin['SMILES'])

smiles = df_smiles_origin['SMILES'].values

smiles_num_array = []
for smile in smiles:
    single_drug = smiles_language.smiles_to_token_indexes(smile)
    single_drug_num_array = np.array(single_drug)
    smiles_num_array.append(single_drug_num_array)

df_smiles = pd.DataFrame(smiles_num_array)
df_smiles.insert(0, 'drug', df_smiles_origin['DRUG_NAME'])


def getTrainTestDataSet(fold):
    path_test = '../../data/10_fold_data/mixed/MixedSet_test_Fold{}.csv'
    path_train = '../../data/10_fold_data/mixed/MixedSet_train_Fold{}.csv'
    # 制作训练集
    df_train = pd.read_csv(path_train.format(fold), index_col=0)
    df_train = pd.merge(df_train, df_GEP, on='cell_line')
    df_train = pd.merge(df_train, df_smiles, on='drug')
    df_train_X = df_train.iloc[:, 3:]
    df_train_y = df_train['IC50']
    # 制作测试集
    df_test = pd.read_csv(path_test.format(fold), index_col=0)
    df_test = pd.merge(df_test, df_GEP, on='cell_line')
    df_test = pd.merge(df_test, df_smiles, on='drug')
    df_test_X = df_test.iloc[:, 3:]
    df_test_y = df_test['IC50']

    return df_train_X, df_test_X, df_train_y, df_test_y


# %%
def main():
    results_dir = os.path.join(os.getcwd(), 'result')

    os.makedirs(results_dir, exist_ok=True)
    X_train, X_val, Y_train, Y_val = getTrainTestDataSet(0)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )

    # Start hyperparameter optimization.
    objective = create_objective(X_train, Y_train, X_val, Y_val, DEVICE)
    study.optimize(objective, n_trials=50)

    # Retrieve the best hyperparameters.
    best_params = study.best_params
    print(f"\nBest hyperparameters for fold {1}:", best_params)

    params_dir = os.path.join(os.getcwd(), 'best_hyp')
    os.makedirs(params_dir, exist_ok=True)
    # Save the best hyperparameters to a file.
    params_file = os.path.join(params_dir, f'precily_best_params.json')
    with open(params_file, 'w') as f:
        json.dump(best_params, f)


    n_folds = 10

    for fold in range(n_folds):
        print(f"\nTraining fold {fold + 1}/{n_folds}")
        X_train, X_val, Y_train, Y_val = getTrainTestDataSet(fold)

        # Read the best hyperparameters.
        params_file = os.path.join('best_hyp', 'precily_best_params.json')
        with open(params_file, 'r') as f:
            best_params = json.load(f)

        # Create the model using the best hyperparameters.
        hidden_layers = [best_params[f'units_{i}'] for i in range(best_params['n_layers'])]
        model = PrecilyModel(INPUT_DIM, hidden_layers, best_params['dropout_rate']).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        early_stopping = EarlyStopping(patience=PATIENCE)

        train_dataset = PrecilyDataset(X_train, Y_train)
        val_dataset = PrecilyDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        for epoch in range(MAX_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            mse, r2, pcc, preds_total, labels_total = validate(model, val_loader, criterion, DEVICE)

            print(f"Fold {fold + 1}, Epoch {epoch + 1}: "
                  f"MSE={mse:.4f} (best={early_stopping.best_mse:.4f}), "
                  f"R2={r2:.4f} (best={early_stopping.best_r2:.4f}), "
                  f"PCC={pcc:.4f} (best={early_stopping.best_pcc:.4f})")

            if early_stopping(mse, r2, pcc, preds_total):
                break

        fold_results = {
            'fold': fold + 1,
            'best_metrics': {
                'mse': float(early_stopping.best_mse),
                'r2': float(early_stopping.best_r2),
                'pcc': float(early_stopping.best_pcc)
            },
            'predictions': early_stopping.best_preds.tolist()
        }

        fold_results_file = os.path.join(results_dir, f'training_fold_{fold + 1}_metrics.json')
        with open(fold_results_file, 'w') as f:
            json.dump(fold_results, f, indent=4)

        # torch.save(model.state_dict(), os.path.join(DIRECTORY, f'precily_cv_{fold + 1}.pth'))

        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()