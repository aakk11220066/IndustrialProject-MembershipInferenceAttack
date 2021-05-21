import torch.nn as nn
import torch
from Models import MLP
from Configuration import LAYER0_SYNC_LOSS_WEIGHT, LAYER2_SYNC_LOSS_WEIGHT


class EntropyAndSyncLoss(nn.Module):
    def __init__(self, attack_model, target_model):
        super(EntropyAndSyncLoss, self).__init__()
        self.attack_model = attack_model
        self.target_model = target_model


    def forward(self, y_pred, y_true, features):
        # Compute attack and target model per-layer pre-activations
        # TODO: optimize by extracting these pre-activations from original model forward-prop instead of recomputing them
        attack_model_results, target_model_results = [], []
        attack_model_results.append(self.attack_model.layers[0](features))
        attack_model_results.append(self.attack_model.layers[2](self.attack_model.layers[1](attack_model_results[0])))
        with torch.no_grad(): # To prevent training from propagating back to the target model, which must be left untouched
            target_model_results.append(self.target_model.layers[0](features))
            target_model_results.append(self.target_model.layers[2](self.target_model.layers[1](target_model_results[0])))

        return nn.CrossEntropyLoss()(y_pred, y_true) + \
            LAYER0_SYNC_LOSS_WEIGHT*nn.MSELoss()(attack_model_results[0], target_model_results[0]) + \
            LAYER2_SYNC_LOSS_WEIGHT*nn.MSELoss()(attack_model_results[1], target_model_results[1])
