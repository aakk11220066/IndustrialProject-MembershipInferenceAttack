import torch.nn as nn
from Models import MLP
from Configuration import LAYER0_SYNC_LOSS_WEIGHT, LAYER2_SYNC_LOSS_WEIGHT


class EntropyAndSyncLoss(nn.Module):
    def __init__(self):
        super(EntropyAndSyncLoss, self).__init__()


    def forward(self, y_pred, y_true, target_model_results, attack_model_results):
        return nn.CrossEntropyLoss()(y_pred, y_true) + \
               LAYER0_SYNC_LOSS_WEIGHT*nn.MSELoss()(attack_model_results[0], target_model_results[0]) + \
               LAYER2_SYNC_LOSS_WEIGHT*nn.MSELoss()(attack_model_results[1], target_model_results[1])
