from torch import Tensor
from BayesAttack import BayesAttackModel
from Models import LinearModel, DisplacementNet
from Configuration import NUM_EPOCHS
from Trainer import get_conv_trainer

class GeneralAttackModel(BayesAttackModel):
    def __init__(self,
                 target_model: LinearModel,
                 attack_train_features: Tensor, attack_train_labels: Tensor):
        super().__init__(
            target_model=target_model,
            attack_train_features=attack_train_features, attack_train_labels=attack_train_labels
        )


    def get_attack_params(self, target_model, proxy_model):
        self.weights_displacer = DisplacementNet()
        get_conv_trainer(model=self.weights_displacer).fit(self.attack_train_features, self.attack_train_labels,
                                                           num_epochs=NUM_EPOCHS)
        attack_model = self.weights_displacer(
            shadow_weights=target_model.layers[0].weight,
            proxy_weights=proxy_model.layers[0].weight,
            shadow_biases=target_model.layers[0].bias,
            proxy_biases=proxy_model.layers[0].bias
        )
        return attack_model.layers[0].weight, attack_model.layers[0].bias
