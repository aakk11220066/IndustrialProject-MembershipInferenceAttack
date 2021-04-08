import torch.nn as nn
from Configuration import NUM_CLASSES, NUM_CLASS_FEATURES


class LinearAttackedModel(nn.Module):
    def __init__(self):
        super(LinearAttackedModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=NUM_CLASS_FEATURES, out_features=NUM_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)
