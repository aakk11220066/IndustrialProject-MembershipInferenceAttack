from torch import Tensor
from MLPDiscriminator import MLPDiscriminatorModel
from Models import MLP, DisplacementNet
from Configuration import NUM_EPOCHS
from Trainer import get_conv_trainer

class GeneralAttackModel(MLPDiscriminatorModel):
    """
    Generalization of BayesAttack in which get_attack_params is changed to use a 1x1 convolution to combine the weights
    and biases of the proxy and target models (a particular case of which is Gaussian distribution) to allow leeway for
    non-Gaussian data distribution
    """
    def __init__(self,
                 target_model: MLP,
                 attack_train_features: Tensor, attack_train_labels: Tensor,
                 test_features, test_labels):
        super().__init__(
            target_model=target_model,
            attack_train_features=attack_train_features, attack_train_labels=attack_train_labels,
            test_features=test_features, test_labels=test_labels
        )


    def get_attack_params(self, target_model, proxy_model, test_features, test_labels):
        """
        Performs convolution between the target model and proxy model to produce discriminator model that produces
        confidence levels for membership of datapoint in target model training set.  Train with siamese pair loss, just
        as in MLP attack (however here there is no need for synchronization loss, due to implicit synchronization
        achieved by using linear model.
        Returns discriminator model weights
        """
        self.weights_displacer = DisplacementNet() # init DisplacementNet
        get_conv_trainer(model=self.weights_displacer, target_model=target_model,
                         test_features=test_features, test_labels=test_labels).fit(
            self.attack_train_features,
            self.attack_train_labels,
            num_epochs=NUM_EPOCHS
        ) # train DisplacementNet
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
        ) # apply DisplacementNet to target and proxy models to produce linear discriminator model
        return layer0_attack_weights.squeeze(dim=0), layer0_attack_bias.squeeze(dim=0), \
               layer2_attack_weights.squeeze(dim=0), layer2_attack_bias.squeeze(dim=0)
