from torch import Tensor
from MLPDiscriminator import MLPDiscriminatorModel
from Models import MLP, DisplacementNet
from Configuration import NUM_EPOCHS
from Trainer import get_conv_trainer

class GeneralAttackModel(MLPDiscriminatorModel):
    def __init__(self,
                 target_model: MLP,
                 attack_train_features: Tensor, attack_train_labels: Tensor):
        super().__init__(
            target_model=target_model,
            attack_train_features=attack_train_features, attack_train_labels=attack_train_labels
        )


    def get_attack_params(self, target_model, proxy_model):
        self.weights_displacer = DisplacementNet()
        get_conv_trainer(model=self.weights_displacer, target_model=target_model).fit(
            self.attack_train_features,
            self.attack_train_labels,
            num_epochs=NUM_EPOCHS
        )
        layer0_attack_weights, layer0_attack_bias, layer2_attack_weights, layer2_attack_bias = \
            self.weights_displacer(
                layer0_shadow_weights=target_model.layers[0].weight.unsqueeze(dim=0),
                layer0_proxy_weights=proxy_model.layers[0].weight.unsqueeze(dim=0),
                layer0_shadow_biases=target_model.layers[0].bias.unsqueeze(dim=0),
                layer0_proxy_biases=proxy_model.layers[0].bias.unsqueeze(dim=0),
                layer2_shadow_weights = target_model.layers[2].weight.unsqueeze(dim=0),
                layer2_proxy_weights = proxy_model.layers[2].weight.unsqueeze(dim=0),
                layer2_shadow_biases = target_model.layers[2].bias.unsqueeze(dim=0),
                layer2_proxy_biases = proxy_model.layers[2].bias.unsqueeze(dim=0)
        )
        return layer0_attack_weights.squeeze(dim=0), layer0_attack_bias.squeeze(dim=0), \
               layer2_attack_weights.squeeze(dim=0), layer2_attack_bias.squeeze(dim=0)

#DELETE THIS COMMENT
