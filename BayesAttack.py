import torch
import torch.nn as nn
from Models import LinearModel
from Configuration import NUM_EPOCHS, get_linear_trainer


class BayesAttackModel(LinearModel):
    """
    Mathematical attack on linear target model under assumption of Gaussian distribution
    """

    def __init__(self,
                 target_model: LinearModel,
                 proxy_train_features: torch.Tensor, proxy_train_labels: torch.Tensor):
        super().__init__(activation=nn.Sigmoid())
        proxy_model = LinearModel()
        get_linear_trainer(proxy_model).fit(proxy_train_features, proxy_train_labels, num_epochs=NUM_EPOCHS)

        attack_weights, attack_bias = self.get_attack_params(target_model=target_model, proxy_model=proxy_model)

        self.layers[0].weight = nn.Parameter(attack_weights)
        self.layers[0].bias = nn.Parameter(attack_bias)

    def get_attack_params(self, target_model, proxy_model):
        attack_weights = target_model.layers[0].weight - proxy_model.layers[0].weight
        attack_bias = target_model.layers[0].bias - proxy_model.layers[0].bias
        return attack_weights, attack_bias

    def forward(self, x, y):
        """
        :param x: datapoint x, shape: (BATCH_SIZE, NUM_FEATURES)
        :param y: datapoint label, shape(BATCH_SIZE,)
        :return: (boolean) datapoint in training set?
        """
        return super().forward(x).gather(dim=1, index=y.unsqueeze(dim=1)) > 0.5