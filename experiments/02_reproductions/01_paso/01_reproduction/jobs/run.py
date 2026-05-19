"""Reproduce PASO's drug-blind r=0.745 using PASO's original code.

Mirrors train_PASO_Kfold_double_omics.py with:
  - OmicsDrugSensitivityDataset_GEP_MUT  (not GEP_CNV)
  - PASO_GEP_MUT model
  - data/10_fold_data/drug_blind/ splits
  - best-epoch selected by TEST-set Pearson (PASO's snooping protocol)

Run via: uv run --with pytoda==1.1.3 python3 jobs/run.py
"""

import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parents[5]   # multi-onco/
PASO_DIR = ROOT / "external" / "PASO"
EXP_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(PASO_DIR))

# PASO imports — requires pytoda==1.1.3 on PYTHONPATH
from utils.PASO_DrugSensitivityDataset_GEP_MUT import OmicsDrugSensitivityDataset_GEP_MUT  # noqa: E402
from utils.hyperparams import OPTIMIZER_FACTORY  # noqa: E402
from utils.loss_functions import pearsonr  # noqa: E402
from utils.utils import get_device, get_log_molar  # noqa: E402
from models.PASO.models import MODEL_FACTORY  # noqa: E402
from pytoda.smiles.smiles_language import SMILESTokenizer  # noqa: E402

# ── paths ────────────────────────────────────────────────────────────────────
DATA = PASO_DIR / "data"
DRUG_BLIND_PFX  = str(DATA / "10_fold_data" / "drug_blind" / "DrugBlind_")
GEP_FILE        = str(DATA / "GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv")
MUT_FILE        = str(DATA / "MUT_Cardinality_Analysis_of_Variance_C2_KEGG_MEDICUS.csv")
SMI_FILE        = str(DATA / "ccle-gdsc.smi")
GENE_FILE       = str(DATA / "MUDICUS_Omic_619_pathways.pkl")
SMILES_LANG     = str(DATA / "smiles_language" / "tokenizer_customized")
PARAMS_FILE     = str(DATA / "best_hyp" / "proposed_model" / "PASO_GEP_MUT.json")
MODEL_OUT_DIR   = str(PASO_DIR / "models" / "PASO" / "result" / "reproduce_DrugBlind_GEP_MUT")
REPORT_DATA     = EXP_DIR / "report" / "data" / "results.json"

N_FOLDS  = 10
N_EPOCHS = 200
BATCH    = 512


def main():
    # Load hyperparameters
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    params.update({"batch_size": BATCH, "epochs": N_EPOCHS, "num_workers": 4, "fold": N_FOLDS})
    print("Params:", params)

    # SMILES tokenizer
    smiles_language = SMILESTokenizer.from_pretrained(SMILES_LANG)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=True,
        padding_length=smiles_language.max_token_sequence_length,
    )
    test_smiles_language = deepcopy(smiles_language)
    for lang, augment in [(smiles_language, params.get("augment_smiles", False)),
                          (test_smiles_language, False)]:
        lang.set_smiles_transforms(
            augment=augment,
            canonical=params.get("smiles_canonical", False),
            kekulize=params.get("smiles_kekulize", False),
            all_bonds_explicit=params.get("smiles_bonds_explicit", False),
            all_hs_explicit=params.get("smiles_all_hs_explicit", False),
            remove_bonddir=params.get("smiles_remove_bonddir", False),
            remove_chirality=params.get("smiles_remove_chirality", False),
            selfies=params.get("selfies", False),
            sanitize=params.get("selfies", False),
        )

    with open(GENE_FILE, "rb") as f:
        pathway_list = pickle.load(f)

    device = get_device()
    fold_results = []

    for fold in range(N_FOLDS):
        print(f"\n====== Fold [{fold+1}/{N_FOLDS}] ======")
        fold_dir = os.path.join(MODEL_OUT_DIR, f"Fold{fold+1}")
        os.makedirs(os.path.join(fold_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "results"), exist_ok=True)

        def make_dataset(split, lang):
            return OmicsDrugSensitivityDataset_GEP_MUT(
                drug_sensitivity_filepath=DRUG_BLIND_PFX + f"{split}_Fold{fold}.csv",
                smiles_filepath=SMI_FILE,
                gep_filepath=GEP_FILE,
                mut_filepath=MUT_FILE,
                gep_standardize=params.get("gep_standardize", True),
                mut_standardize=params.get("mut_standardize", False),
                smiles_language=lang,
                drug_sensitivity_min_max=params.get("drug_sensitivity_min_max", True),
                iterate_dataset=False,
            )

        train_ds = make_dataset("train", smiles_language)
        test_ds  = make_dataset("test",  test_smiles_language)
        min_ic50 = test_ds.drug_sensitivity_processing_parameters["parameters"]["min"]
        max_ic50 = test_ds.drug_sensitivity_processing_parameters["parameters"]["max"]

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
            num_workers=params.get("num_workers", 4))
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=BATCH, shuffle=False, drop_last=True,  # matches PASO original
            num_workers=params.get("num_workers", 4))

        params.update({
            "number_of_genes": len(pathway_list),
            "smiles_vocabulary_size": smiles_language.number_of_tokens,
            "drug_sensitivity_processing_parameters":
                train_ds.drug_sensitivity_processing_parameters,
            "model_fn": "PASO_GEP_MUT",
        })
        model = MODEL_FACTORY["PASO_GEP_MUT"](params).to(device)
        model._associate_language(smiles_language)

        optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](
            model.parameters(), lr=params.get("lr", 0.001))

        max_pearson, best_epoch = 0.0, 0

        for epoch in range(1, N_EPOCHS + 1):
            # --- train ---
            model.train()
            for smiles, omic_gep, omic_mut, y in train_loader:
                y_hat, _ = model(
                    torch.squeeze(smiles.to(device)),
                    omic_gep.to(device), omic_mut.to(device))
                loss = model.loss(y_hat, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- evaluate on TEST (PASO's protocol: checkpoint by test Pearson) ---
            model.eval()
            log_pres, log_labels = [], []
            with torch.no_grad():
                for smiles, omic_gep, omic_mut, y in test_loader:
                    _, pred_dict = model(
                        torch.squeeze(smiles.to(device)),
                        omic_gep.to(device), omic_mut.to(device))
                    log_pre = pred_dict.get("log_micromolar_IC50")
                    log_y = get_log_molar(y, ic50_max=max_ic50, ic50_min=min_ic50)
                    log_pres.append(log_pre)
                    log_labels.append(log_y)

            preds  = torch.cat([p.cpu() for ps in log_pres for p in ps])
            labels = torch.cat([l.cpu() for ls in log_labels for l in ls])
            r = float(pearsonr(torch.Tensor(preds), torch.Tensor(labels)))
            print(f"  Epoch {epoch:3d}: test_pearson={r:.4f}")

            # Save checkpoint when test Pearson improves (PASO's snooping mechanism)
            if r > max_pearson:
                max_pearson = r
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(fold_dir, "weights", "best_pearson.pt"))

        result = {"fold": fold + 1, "best_test_r": max_pearson, "best_epoch": best_epoch}
        fold_results.append(result)
        print(f"  => Fold {fold+1} best: r={max_pearson:.4f} at epoch {best_epoch}")
        with open(os.path.join(fold_dir, "results", "pearson.json"), "w") as f:
            json.dump(result, f, indent=2)

    # ── summarise ────────────────────────────────────────────────────────────
    best_r_per_fold = [r["best_test_r"] for r in fold_results]
    summary = {
        "fold_best_test_r":  best_r_per_fold,
        "mean":              float(np.mean(best_r_per_fold)),
        "std":               float(np.std(best_r_per_fold)),
        "best_fold_r":       float(max(best_r_per_fold)),
        "best_fold_index":   int(np.argmax(best_r_per_fold)) + 1,
        "fold_details":      fold_results,
    }
    REPORT_DATA.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DATA.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Results saved to {REPORT_DATA}")
    print(f"10-fold mean: {summary['mean']:.4f} ± {summary['std']:.4f}")
    print(f"Best fold:    {summary['best_fold_r']:.4f} (fold {summary['best_fold_index']})")
    print(f"Expected:     best fold ≈ 0.745 (PASO paper headline; fold identity confirmed after run)")


if __name__ == "__main__":
    main()
