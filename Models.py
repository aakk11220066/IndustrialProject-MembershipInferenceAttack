import torch.nn as nn
import torch
from Configuration import NUM_CLASSES, NUM_CLASS_FEATURES


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
        self.conv = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, shadow_model: nn.Module, proxy_model: nn.Module):
        shadow_weight = shadow_model.layers[0].weight
        shadow_bias = shadow_model.layers[0].bias
        proxy_weight = proxy_model.layers[0].weight
        proxy_bias = proxy_model.layers[0].bias

        attack_weight = self.conv(torch.stack((shadow_weight, proxy_weight)))
        attack_bias = self.conv(torch.stack((shadow_bias, proxy_bias))).squeeze()

        prediction_maker = LinearModel(num_classes=2)
        prediction_maker.layers[0].weight = attack_weight
        prediction_maker.layers[0].bias = attack_bias
        return prediction_maker
