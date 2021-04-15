import torch.nn as nn
from Configuration import NUM_CLASSES, NUM_CLASS_FEATURES


class LinearModel(nn.Module):
    def __init__(self, activation=nn.Softmax(dim=1)):
        super(LinearModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=NUM_CLASS_FEATURES, out_features=NUM_CLASSES),
            activation
        )

    def forward(self, x):
        return self.layers(x)
