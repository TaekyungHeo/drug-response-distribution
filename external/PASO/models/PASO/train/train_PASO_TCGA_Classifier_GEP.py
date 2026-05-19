
import json
import os
import pickle
from copy import deepcopy
from time import time
import numpy as np
from models.PASO.models import MODEL_FACTORY
import torch
from utils.TCGADataset_GEP import TCGADataset_GEP
from utils.hyperparams import OPTIMIZER_FACTORY
from utils.loss_functions import pearsonr, r2_score, calculate_aucpr
from utils.utils import get_device
from pytoda.smiles.smiles_language import SMILESTokenizer


def main(
    train_sensitivity_filepath,
    test_sensitivity_filepath,
    gep_filepath,
    smi_filepath,
    gene_filepath,
    smiles_language_filepath,
    model_path,
    params_filepath,
    training_name
):

    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))
        params.update(
            {
                "batch_size": 365,
                "epochs": 50,
                "num_workers": 0,
            }
        )
    print(params)
    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
    with open(os.path.join(model_dir, "TCGA_classifier_best_aucpr_GEP.json"), "w") as fp:
        json.dump(params, fp, indent=4)

    # Prepare the dataset
    print("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=params.get("padding", True),
        padding_length=smiles_language.max_token_sequence_length,
        # padding_length=params.get("smiles_padding_length", None),
    )
    test_smiles_language = deepcopy(smiles_language)
    smiles_language.set_smiles_transforms(
        augment=params.get("augment_smiles", False),
        canonical=params.get("smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )
    test_smiles_language.set_smiles_transforms(
        augment=False,
        canonical=params.get("test_smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )

    # Load the gene list
    with open(gene_filepath, "rb") as f:
        pathway_list = pickle.load(f)

    # Load the datasets
    train_dataset = TCGADataset_GEP(
        drug_sensitivity_filepath=train_sensitivity_filepath,
        smiles_filepath=smi_filepath,
        gep_filepath=gep_filepath,
        gep_standardize=params.get("gep_standardize", False),
        smiles_language=smiles_language,
        drug_sensitivity_min_max=False,
        iterate_dataset=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params.get("num_workers", 4),
    )
    test_dataset = TCGADataset_GEP(
        drug_sensitivity_filepath=test_sensitivity_filepath,
        smiles_filepath=smi_filepath,
        gep_filepath=gep_filepath,
        gep_standardize=params.get("gep_standardize", False),
        smiles_language=smiles_language,
        drug_sensitivity_min_max=False,
        iterate_dataset=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=params.get("num_workers", 4),
    )
    print(
        f"Training dataset has {len(train_dataset)} samples, test set has "
        f"{len(test_dataset)}."
    )

    device = get_device()

    save_top_model = os.path.join(model_dir, "weights/{}_{}_{}.pt")
    params.update(
        {  # yapf: disable
            "number_of_genes": len(pathway_list),
            "smiles_vocabulary_size": smiles_language.number_of_tokens,
            "drug_sensitivity_processing_parameters": train_dataset.drug_sensitivity_processing_parameters,
            "gene_expression_processing_parameters": {},
        }
    )
    model_name = params.get("model_fn", "PASO_GEP_Classifier")
    model = MODEL_FACTORY[model_name](params).to(device)
    model._associate_language(smiles_language)


    if os.path.isfile(os.path.join(model_dir, "weights", f"best_mse_{model_name}.pt")):
        print("Found existing model, restoring now...")
        model.load(os.path.join(model_dir, "weights", f"best_mse_{model_name}.pt"))

        with open(os.path.join(model_dir, "results", "mse.json"), "r") as f:
            info = json.load(f)
            min_loss = float(info["test_loss"].strip("tensor()"))
            max_aucpr = float(info["aucpr"].strip("tensor()"))
    else:
        min_loss, min_rmse, max_aucpr = 100, 1000, 0

    # Define optimizer
    optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](
        model.parameters(), lr=params.get("lr", 0.01)
    )


    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({"number_of_parameters": num_params})
    print(f"Number of parameters {num_params}")

    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, "TCGA_classifier_best_aucpr_GEP.json"), "w") as fp:
        json.dump(params, fp)

    # Start training
    print("Training about to start...\n")
    t = time()

    model.save(save_top_model.format("epoch", "0", model_name))

    for epoch in range(params["epochs"]):

        model.train()
        print(params_filepath.split("/")[-1])
        print(f"== Epoch [{epoch+1}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, gep, y) in enumerate(train_loader):
            y_hat, pred_dict = model(torch.squeeze(smiles.to(device)), gep.to(device))
            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-6)
            optimizer.step()
            train_loss += loss.item()

        print(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )
        t = time()

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            response_pres = []
            for ind, (smiles, gep, y) in enumerate(test_loader):
                y_hat, pred_dict = model(
                    torch.squeeze(smiles.to(device)), gep.to(device)
                )
                response_pre = pred_dict.get("response_pres")
                response_pres.append(response_pre)
                predictions.append(y_hat)
                labels.append(y)
                loss = model.loss(response_pre, y.to(device))
                test_loss += loss.item()

        predictions = torch.cat([p.cpu() for preds in predictions for p in preds])
        labels = torch.cat([l.cpu() for label in labels for l in label])
        # aucpr
        AUCPR_a = calculate_aucpr(labels, predictions)
        # loss
        test_loss_a = test_loss / len(test_loader)
        print(
            f"\t **** TESTING ****   Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {test_loss_a:.5f}, "
            f"AUCPR: {AUCPR_a:.5f}"
        )

        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, model_name))
            with open(os.path.join(model_dir, "results", metric + ".json"), "w") as f:
                json.dump(info, f)
            np.save(
                os.path.join(model_dir, "results", metric + "_preds.npy"),
                np.vstack([predictions, labels]),
            )
            if typ == "best":
                print(
                    f'\t New best performance in "{metric}"'
                    f" with value : {val:.7f} in epoch: {epoch}"
                )

        def update_info():
            return {
                "test_loss": str(min_loss),
                "aucpr": str(max_aucpr),
                "predictions": [float(p) for p in predictions],
            }

        if test_loss_a < min_loss:
            min_loss = test_loss_a
            info = update_info()
            save(save_top_model, "mse", "best", min_loss)
            ep_loss = epoch

        if AUCPR_a > max_aucpr:
            max_aucpr = AUCPR_a
            info = update_info()
            save(save_top_model, "aucpr", "best", max_aucpr)
            ep_aucpr = epoch

        # if (epoch + 1) % params.get("save_model", 100) == 0:
        #     save(save_top_model, "epoch", str(epoch))
    print(
        f"Overall best performances are: \n \t"
        f"Loss = {min_loss:.4f} in epoch {ep_loss} "
        f"\t AUCPR = {max_aucpr:.4f} in epoch {ep_aucpr}"

    )
    save(save_top_model, "training", "done")
    print("Done with training, models saved, shutting down.")


if __name__ == "__main__":

    train_sensitivity_filepath = '../../../data/data_reproduce/TCGA_drug_sensitivity_MixedSet_train2.csv'
    test_sensitivity_filepath = '../../../data/data_reproduce/TCGA_drug_sensitivity_MixedSet_test2.csv'
    gep_filepath = '../../../data/TCGA_GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'

    smi_filepath = '../../../data/ccle-gdsc-TCGA.smi'
    gene_filepath = '../../../data/MUDICUS_Omic_619_pathways.pkl'
    smiles_language_filepath = '../../../data/smiles_language/tokenier_customized_TCGA'
    model_path = 'result/model'
    params_filepath = '../../../data/best_hyp/proposed_model/PASO_GEP_Classifier.json'
    # training_name = 'TCGA_Classifier_Unique_Trail3'
    training_name = 'TCGA_Classifier_Trail6'
    # run the training
    main(
        train_sensitivity_filepath,
        test_sensitivity_filepath,
        gep_filepath,
        smi_filepath,
        gene_filepath,
        smiles_language_filepath,
        model_path,
        params_filepath,
        training_name
    )
