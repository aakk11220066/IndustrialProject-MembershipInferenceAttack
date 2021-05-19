import torch
import torch.nn as nn
from Models import MLP
from Configuration import NUM_EPOCHS
from Trainer import get_regular_model_trainer

class MLPDiscriminatorModel(MLP):
    """
    Mathematical attack on linear target model under assumption of Gaussian distribution
    """

    def __init__(self,
                 target_model: MLP,
                 attack_train_features: torch.Tensor, attack_train_labels: torch.Tensor):
        super().__init__(activation=nn.Sigmoid())
        self.attack_train_features = attack_train_features
        self.attack_train_labels = attack_train_labels

        proxy_model = MLP()
        get_regular_model_trainer(proxy_model).fit(attack_train_features, attack_train_labels, num_epochs=NUM_EPOCHS)

        layer0_attack_weights, layer0_attack_bias, layer2_attack_weights, layer2_attack_bias = \
            self.get_attack_params(target_model=target_model, proxy_model=proxy_model)

        self.layers[0].weight = nn.Parameter(layer0_attack_weights)
        self.layers[0].bias = nn.Parameter(layer0_attack_bias)
        self.layers[2].weight = nn.Parameter(layer2_attack_weights)
        self.layers[2].bias = nn.Parameter(layer2_attack_bias)


    def get_attack_params(self, target_model, proxy_model):
        raise NotImplementedError()


    def forward(self, x, y):
        """
        :param x: datapoint x, shape: (BATCH_SIZE, NUM_FEATURES)
        :param y: datapoint label, shape(BATCH_SIZE,)
        :return: (boolean) datapoint in training set?
        """
        return super().forward(x).gather(dim=1, index=y.unsqueeze(dim=1)) > 0.5

#DELETE THIS COMMENT