import torch.nn as nn
import torch
from Models import MLP
from Configuration import LAYER0_SYNC_LOSS_WEIGHT, LAYER2_SYNC_LOSS_WEIGHT


class SynchronizationLoss(nn.Module):
    def __init__(self, attack_model, target_model):
        super(SynchronizationLoss, self).__init__()
        self.attack_model = attack_model
        self.target_model = target_model
        self.loss = nn.MSELoss()


    def forward(self, y_pred, y_true, features):
        # Compute attack and target model per-layer pre-activations
        # TODO: optimize by extracting these pre-activations from original model forward-prop instead of recomputing them
        # y_pred and y_true go unused, only there in order to fit the abstract structure
        attack_model_results, target_model_results = [], []
        attack_model_results.append(self.attack_model.layers[0](features))
        attack_model_results.append(self.attack_model.layers[2](self.attack_model.layers[1](attack_model_results[0])))
        with torch.no_grad(): # To prevent training from propagating back to the target model, which must be left untouched
            target_model_results.append(self.target_model.layers[0](features))
            target_model_results.append(self.target_model.layers[2](self.target_model.layers[1](target_model_results[0])))

        return \
            LAYER0_SYNC_LOSS_WEIGHT*self.loss(attack_model_results[0], target_model_results[0]) + \
            LAYER2_SYNC_LOSS_WEIGHT*self.loss(attack_model_results[1], target_model_results[1])


class EntropyAndSyncLoss(nn.Module):
    def __init__(self, attack_model, target_model):
        super(EntropyAndSyncLoss, self).__init__()
        self.sync_loss = SynchronizationLoss(attack_model=attack_model, target_model=target_model)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true, features):
        return self.loss(y_pred, y_true) + self.sync_loss(y_pred=y_pred, y_true=y_true, features=features)
