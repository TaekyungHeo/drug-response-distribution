import json
import pickle
from copy import deepcopy
import pandas as pd
import torch
from models.PASO.models import MODEL_FACTORY
from pytoda.smiles.smiles_language import SMILESTokenizer
from utils.TCGADataset_GEP import TCGADataset_GEP
from utils.loss_functions import calculate_aucpr
from utils.utils import get_device


def main(
    test_sensitivity_filepath,
    gep_filepath,
    smi_filepath,
    gene_filepath,
    smiles_language_filepath,
    model_path,
    params_filepath,
):

    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))
        params.update(
            {
                "num_workers": 0
            }
        )
    print(params)

    # Prepare the dataset
    print("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=params.get("padding", True),
        padding_length=smiles_language.max_token_sequence_length,
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
    test_dataset = TCGADataset_GEP(
        drug_sensitivity_filepath=test_sensitivity_filepath,
        smiles_filepath=smi_filepath,
        gep_filepath=gep_filepath,
        gep_standardize=params.get("gep_standardize", False),
        smiles_language=smiles_language,
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
        f'Test dataset has {len(test_dataset)} samples with {len(test_loader)} batches'
    )

    device = get_device()

    print(
        f'model is {device}'
    )

    model_name = params.get("model_fn", "PASO_GEP_Classifier")
    model = MODEL_FACTORY[model_name](params).to(device)
    model._associate_language(smiles_language)

    try:
        print(f'Attempting to restore model from {model_path}...')
        model.load(model_path, map_location=device)
    except Exception:
        raise ValueError(f'Error in restoring model from {model_path}!')

    model.eval()
    with torch.no_grad():
        test_loss = 0
        predictions = []
        labels = []
        response_pres = []
        gene_attentions = []
        # cnv_attentions = []
        # mut_attentions = []
        smiles_attentions_geps = []
        # smiles_attentions_cnvs = []
        # smiles_attentions_muts = []
        for ind, (smiles, gep, y) in enumerate(test_loader):
            y_hat, pred_dict = model(
                torch.squeeze(smiles.to(device)), gep.to(device)
            )
            response_pre = pred_dict.get("response_pres")
            response_pres.append(response_pre)
            predictions.append(y_hat)
            labels.append(y)
            gene_attention = pred_dict.get("gene_attention")
            # 通过取均值的方法将gene_attention维度从[192,619,5]->[192,619]
            gene_attention = torch.mean(gene_attention, dim=2)
            gene_attentions.append(gene_attention)
            # cnv_attention = pred_dict.get("cnv_attention")
            # cnv_attention = torch.mean(cnv_attention, dim=2)
            # cnv_attentions.append(cnv_attention)
            # mut_attention = pred_dict.get("mut_attention")
            # mut_attention = torch.mean(mut_attention, dim=2)
            # mut_attentions.append(mut_attention)
            smiles_attention_gep = pred_dict.get("smiles_attention_gep")
            smiles_attention_gep = torch.mean(smiles_attention_gep, dim=2)
            smiles_attentions_geps.append(smiles_attention_gep)
            # smiles_attention_cnv = pred_dict.get("smiles_attention_cnv")
            # smiles_attention_cnv = torch.mean(smiles_attention_cnv, dim=2)
            # smiles_attentions_cnvs.append(smiles_attention_cnv)
            # smiles_attention_mut = pred_dict.get("smiles_attention_mut")
            # smiles_attention_mut = torch.mean(smiles_attention_mut, dim=2)
            # smiles_attentions_muts.append(smiles_attention_mut)
            loss = model.loss(response_pre, y.to(device))
            test_loss += loss.item()

    # on the logIC50 scale
    predictions = torch.cat([p.cpu() for preds in predictions for p in preds])
    labels = torch.cat([l.cpu() for label in labels for l in label])
    # aucpr
    AUCPR_a = calculate_aucpr(labels, predictions)
    # loss
    test_loss_a = test_loss / len(test_loader)
    print(
        f"\t **** TESTING **** \n"
            f"loss: {test_loss_a:.5f}, "
            f"AUCPR: {AUCPR_a:.5f}"
    )

    gene_attentions = torch.cat([gene_attentions[i] for i in range(len(gene_attentions))])
    smiles_attentions_geps = torch.cat([smiles_attentions_geps[i] for i in range(len(smiles_attentions_geps))])

    gene_attentions = gene_attentions.cpu().numpy()
    smiles_attentions_geps = smiles_attentions_geps.cpu().numpy()

    gene_attentions = pd.DataFrame(gene_attentions)
    smiles_attentions_geps = pd.DataFrame(smiles_attentions_geps)

    gene_attentions.to_csv('attention_result/TransMCA_Pathway_Attention_lung_CellBlind_GEP_V2_RAW.csv',index=False)
    smiles_attentions_geps.to_csv('attention_result/TransMCA_Smiles_Attention_lung_CellBlind_GEP_V2_RAW.csv',index=False)

if __name__ == "__main__":

    test_sensitivity_filepath = '../../../data/data_reproduce/TCGA_drug_sensitivity_MixedSet_test.csv'
    gep_filepath = '../../../data/TCGA_GEP_Wilcoxon_Test_Analysis_Log10_P_value_C2_KEGG_MEDICUS.csv'
    smi_filepath = '../../../data/ccle-gdsc-TCGA.smi'
    pathway_filepath = '../../../data/MUDICUS_Omic_619_pathways.pkl'
    smiles_language_filepath = '../../../data/smiles_language/tokenier_customized_TCGA'
    model_path = 'trained_models/TCGA_classifier_best_aucpr_GEP.pt'
    params_filepath = 'trained_models/TCGA_classifier_best_aucpr_GEP.json'
    # training_name = 'TRANS_MCA_GEP(Log10_P_value)_CNV(Cardinality_Analysis)_MUT_MEDICUS619'
    # run the training
    main(
        test_sensitivity_filepath,
        gep_filepath,
        smi_filepath,
        pathway_filepath,
        smiles_language_filepath,
        model_path,
        params_filepath,
        # training_name
    )
