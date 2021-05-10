from typing import List

import torch.nn as nn
import torch
from Configuration import NUM_CLASSES, NUM_CLASS_FEATURES

from torch.nn import functional as F

params = [
    torch.Tensor(32, 1).uniform_(-1., 1.).requires_grad_(),
    torch.Tensor(32).zero_().requires_grad_()
]




class LinearModel(nn.Module):
    def __init__(self, activation=nn.Softmax(dim=1), num_class_features=NUM_CLASS_FEATURES, num_classes=NUM_CLASSES):
        super(LinearModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=num_class_features, out_features=num_classes),
            activation
        )

    def forward(self, x):
        return self.layers(x)


class DisplacementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        #self.b_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, shadow_weights, proxy_weights, shadow_biases, proxy_biases):
        if len(shadow_weights.shape) < 3:
            shadow_weights = shadow_weights.unsqueeze(dim=0)
        if len(proxy_weights.shape) < 3:
            proxy_weights = proxy_weights.unsqueeze(dim=0)
        if len(shadow_biases.shape) < 2:
            shadow_biases = shadow_biases.unsqueeze(dim=0)
        if len(proxy_biases.shape) < 2:
            proxy_biases = proxy_biases.unsqueeze(dim=0)

        attack_weights = self.conv(torch.stack((shadow_weights, proxy_weights), dim=1))
        attack_biases = self.conv(torch.stack((shadow_biases, proxy_biases), dim=1).unsqueeze(dim=-1)).squeeze(dim=-1)
        if len(attack_weights.shape) > 3:
            attack_weights = attack_weights.squeeze(dim=1)
        if len(attack_biases.shape) > 2:
            attack_biases = attack_biases.squeeze(dim=1)

        return attack_weights, attack_biases




