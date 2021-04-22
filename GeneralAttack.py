from torch import Tensor
from BayesAttack import BayesAttackModel
from Models import LinearModel, DisplacementNet
from Configuration import NUM_EPOCHS, get_conv_trainer

class GeneralAttackModel(BayesAttackModel):
    def __init__(self,
                 target_model: LinearModel,
                 train_features: Tensor, train_labels: Tensor):
        super().__init__(
            target_model=target_model,
            proxy_train_features=train_features, proxy_train_labels=train_labels
        )
        self.params_finder = DisplacementNet()

        # Train self.params_finder
        get_conv_trainer(model=self.params_finder).fit(train_features, train_labels, num_epochs=NUM_EPOCHS)


    def get_attack_params(self, target_model, proxy_model):
        return self.weights_displacer(shadow_model=target_model, proxy_model=proxy_model)
