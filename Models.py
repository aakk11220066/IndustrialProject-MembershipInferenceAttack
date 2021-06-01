import torch.nn as nn
import torch
from Configuration import NUM_CLASSES, NUM_CLASS_FEATURES, HIDDEN_DIM


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


class MLP(nn.Module):
    def __init__(self, activation=nn.Softmax(dim=1), num_class_features=NUM_CLASS_FEATURES, num_classes=NUM_CLASSES):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=num_class_features, out_features=HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM, out_features=num_classes),
            activation
        )

    def forward(self, x):
        return self.layers(x)


class DisplacementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.layer2_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        # To prevent ReLU saturation of attack model, we must ensure that attack weights
        # start out non-negative.  But only initialize this way, from here on out we freely train the conv biases
        self.layer0_conv.bias = nn.Parameter(self.layer0_conv.bias.abs())
        self.layer2_conv.bias = nn.Parameter(self.layer2_conv.bias.abs())


    def forward(self, layer0_shadow_weights, layer0_proxy_weights, layer0_shadow_biases, layer0_proxy_biases,
                layer2_shadow_weights, layer2_proxy_weights, layer2_shadow_biases, layer2_proxy_biases):

        layer0_attack_weights = self.layer0_conv(torch.stack((layer0_shadow_weights, layer0_proxy_weights), dim=1))
        layer0_attack_biases = self.layer0_conv(torch.stack((layer0_shadow_biases, layer0_proxy_biases), dim=1)
                                         .unsqueeze(dim=-1)).squeeze(dim=-1).squeeze(dim=1)

        layer2_attack_weights = self.layer2_conv(torch.stack((layer2_shadow_weights, layer2_proxy_weights), dim=1))
        layer2_attack_biases = self.layer2_conv(torch.stack((layer2_shadow_biases, layer2_proxy_biases), dim=1)
                                         .unsqueeze(dim=-1)).squeeze(dim=-1).squeeze(dim=1)

        return layer0_attack_weights, layer0_attack_biases, layer2_attack_weights, layer2_attack_biases




#DELETE THIS COMMENT