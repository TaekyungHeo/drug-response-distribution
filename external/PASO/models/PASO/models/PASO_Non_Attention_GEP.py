import logging
import sys
from collections import OrderedDict

import pytoda
import torch
import torch.nn as nn
from pytoda.smiles.transforms import AugmentTensor

from utils.hyperparams import LOSS_FN_FACTORY, ACTIVATION_FN_FACTORY
from utils.interpret import monte_carlo_dropout, test_time_augmentation
from utils.layers import convolutional_layer, dense_layer, projection_layer
from utils.utils import get_device, get_log_molar

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PASO_GEP_WithOut_Attention(nn.Module):

    def __init__(self, params, *args, **kwargs):

        super(PASO_GEP_WithOut_Attention, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]
        self.min_max_scaling = True if params.get(
            'drug_sensitivity_processing_parameters', {}
        ) != {} else False
        if self.min_max_scaling:
            self.IC50_max = params[
                'drug_sensitivity_processing_parameters'
            ]['parameters']['max']  # yapf: disable
            self.IC50_min = params[
                'drug_sensitivity_processing_parameters'
            ]['parameters']['min']  # yapf: disable

        # Model inputs
        self.smiles_padding_length = params['smiles_padding_length']
        self.number_of_genes = params.get('number_of_genes', 619)
        self.smiles_attention_size = params.get('smiles_attention_size', 64)
        self.gene_attention_size = params.get('gene_attention_size', 1)
        self.molecule_temperature = params.get('molecule_temperature', 1.)
        self.gene_temperature = params.get('gene_temperature', 1.)

        # Model architecture (hyperparameter)
        self.molecule_gep_heads = params.get('molecule_gep_heads', [1, 1, 1, 1])
        self.gene_heads = params.get('gene_heads', [1, 1, 1, 1])
        self.n_heads = params.get('n_heads', 1)
        self.num_layers = params.get('num_layers', 2)
        self.omics_dense_size = params.get('omics_dense_size', 128)
        self.filters = params.get('filters', [64, 64, 64])
        self.kernel_sizes = params.get(
            'kernel_sizes', [
                [3, params['smiles_embedding_size']],
                [5, params['smiles_embedding_size']],
                [11, params['smiles_embedding_size']]
            ]
        )
        self.omic_smiles_dense_size = params.get('omic_smiles_dense_size', 256)
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                'Length of filter and kernel size lists do not match.'
            )
        if len(self.filters) + 2 != len(self.molecule_gep_heads):
            raise ValueError(
                'Length of filter and multihead lists do not match'
            )

        self.hidden_sizes = (
            [
                len(self.molecule_gep_heads) * self.omic_smiles_dense_size
            ] + params.get('stacked_dense_hidden_sizes', [512, 128])
        )

        self.dropout = params.get('dropout', 0.5)
        self.temperature = params.get('temperature', 1.)
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')]

        # Build the model
        self.smiles_embedding = nn.Embedding(
            self.params['smiles_vocabulary_size'],
            self.params['smiles_embedding_size'],
            scale_grad_by_freq=params.get('embed_scale_grad', False)
        )

        encoder = nn.TransformerEncoderLayer(d_model=self.params['smiles_embedding_size'], nhead=self.n_heads, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder, self.num_layers)

        # SMILES Convolutions
        self.convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            batch_norm=params.get('batch_norm', False),
                            dropout=self.dropout
                        ).to(self.device)
                    ) for index, (num_kernel, kernel_size) in
                    enumerate(zip(self.filters, self.kernel_sizes))
                ]
            )
        )
        # smiles
        smiles_hidden_sizes = ([params['smiles_embedding_size']] +
                               [params['smiles_embedding_size']] + self.filters)
        # Project each SMILES. (batchsize,256,smiles_hidden_size) -> (batchsize,256)
        self.smiles_projection_dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'smiles_projection_dense_{}'.format(ind),
                        projection_layer(
                            smiles_hidden_sizes[ind],
                            1,
                        ).to(self.device)
                    ) for ind in range(len(smiles_hidden_sizes))
                ]
            )
        )

        # Construct multiple dense layers based on the length of `smiles_hidden_sizes`, with each layer having different input and output dimensions determined by `smiles_hidden_sizes + number_of_genes`.
        self.omic_smiles_dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.number_of_genes + self.omic_smiles_dense_size,
                            self.omic_smiles_dense_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', False)
                        ).to(self.device)
                    ) for ind in range(len(smiles_hidden_sizes))
                ]
            )
        )

        # Only applied if params['batch_norm'] = True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
        )

    def forward(self, smiles, gep, confidence=False):
        """
        Args:
            smiles (torch.Tensor): of type int and shape: [bs, smiles_padding_length]
            gep (torch.Tensor): of shape `[bs, number_of_genes]`.
            confidence (bool, optional) whether the confidence estimates are
                performed.

        Returns:
            (torch.Tensor, dict): predictions, prediction_dict
            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """

        gep = torch.unsqueeze(gep, dim=-1)
        embedded_smiles = self.smiles_embedding(smiles.to(dtype=torch.int64))

        # Transformer Encoder
        trans_smiles = self.transformer_encoder(embedded_smiles)

        # SMILES Convolutions. Unsqueeze has shape bs x 1 x T x H. 共五个药物尺度
        encoded_smiles = [embedded_smiles] + [trans_smiles] + [
            self.convolutional_layers[ind]
            (torch.unsqueeze(embedded_smiles, 1)).permute(0, 2, 1)
            for ind in range(len(self.convolutional_layers))
        ]
        # SMILES Projection
        smiles_projections = [
            self.smiles_projection_dense_layers[ind](encoded_smiles[ind])
            for ind in range(len(encoded_smiles))
        ]
        encodings = []
        # SMILES Dense
        for layer in range(len(self.omic_smiles_dense_layers)):
            smiles_projections[layer] = torch.cat(
                [smiles_projections[layer], gep], dim=1
            )
            encodings.append(self.omic_smiles_dense_layers[layer](
                smiles_projections[layer].squeeze(-1)
            ))

        encodings = torch.cat(encodings, dim=1)
        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get(
            'batch_norm', False
        ) else encodings
        # NOTE: stacking dense layers as a bottleneck
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            prediction_dict.update({
                'IC50': predictions,
                'log_micromolar_IC50':
                    get_log_molar(
                        predictions,
                        ic50_max=self.IC50_max,
                        ic50_min=self.IC50_min
                    ) if self.min_max_scaling else predictions
            })  # yapf: disable

            if confidence:
                augmenter = AugmentTensor(self.smiles_language)
                epi_conf, epi_pred = monte_carlo_dropout(
                    self,
                    regime='tensors',
                    tensors=(smiles, gep),
                    repetitions=5
                )
                ale_conf, ale_pred = test_time_augmentation(
                    self,
                    regime='tensors',
                    tensors=(smiles, gep),
                    repetitions=5,
                    augmenter=augmenter,
                    tensors_to_augment=0
                )

                prediction_dict.update({
                    'epistemic_confidence': epi_conf,
                    'epistemic_predictions': epi_pred,
                    'aleatoric_confidence': ale_conf,
                    'aleatoric_predictions': ale_pred
                })  # yapf: disable

        elif confidence:
            logger.info('Using confidence in training mode is not supported.')

        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def _associate_language(self, smiles_language):
        """
        Bind a SMILES language object to the model. Is only used inside the
        confidence estimation.

        Arguments:
            smiles_language {[pytoda.smiles.smiles_language.SMILESLanguage]}
            -- [A SMILES language object]

        Raises:
            TypeError:
        """
        if not isinstance(
            smiles_language, pytoda.smiles.smiles_language.SMILESLanguage
        ):
            raise TypeError(
                'Please insert a smiles language (object of type '
                'pytoda.smiles.smiles_language.SMILESLanguage). Given was '
                f'{type(smiles_language)}'
            )
        self.smiles_language = smiles_language

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
