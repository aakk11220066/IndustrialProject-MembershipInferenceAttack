import torch.nn as nn
import torch
from Models import MLP
from Configuration import LAYER0_SYNC_LOSS_WEIGHT_SHADOW, LAYER2_SYNC_LOSS_WEIGHT_SHADOW, LAYER0_SYNC_LOSS_WEIGHT_PROXY, LAYER2_SYNC_LOSS_WEIGHT_PROXY
import matplotlib.pyplot as plt


# For recording test losses and accuracies throughout training
layer0_sync_losses = []
layer2_sync_losses = []
training_accuracies = []
testing_accuracies = []
crossentropy_losses = []

test_features = None
test_labels = None
def init_test_data(_test_features, _test_labels):
    """
    Setup for running evaluation of models on test data
    """
    global test_features, test_labels
    test_features = _test_features
    test_labels = _test_labels

def accuracy(predictions, labels):
    """
    :param predictions: matrix bearing the model's current confidences for membership in each possible class
    :param labels: matrix of data's correct classifications, in one-hot format
    :return: percent of test data evaluated correctly
    """
    confidence_levels = predictions  # (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
    test_labels = labels
    # class_label_predictions = confidence_levels.argmax(dim=1)  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
    #class_label_predictions = torch.stack([1-confidence_levels, confidence_levels], dim=1).multinomial(1).squeeze(dim=1)  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
    class_label_predictions = confidence_levels.round()*1.01
    num_correct = (class_label_predictions == test_labels).sum().item()
    return num_correct / test_labels.shape[0]

def display_losses(model_type: str):
    """
    Plot out the losses and accuracies of the last model training instance.
    :param model_type: Should be "proxy" or "shadow"
    """
    global layer0_sync_losses, layer2_sync_losses, crossentropy_losses, training_accuracies, testing_accuracies
    plt.plot(list(range(len(layer0_sync_losses))), layer0_sync_losses, color="b")
    plt.plot(list(range(len(layer2_sync_losses))), layer2_sync_losses, color="g")
    if model_type == "Proxy":
        plt.plot(list(range(len(crossentropy_losses))), crossentropy_losses, color="r")
    else:
        crossentropy_losses = [0] * len(layer2_sync_losses)
    plt.plot(list(range(len(crossentropy_losses))), [crossentropy_losses[i] + layer2_sync_losses[i] + layer0_sync_losses[i] for i in range(len(crossentropy_losses))], color="y")
    plt.xlabel("Train iteration no.")
    plt.legend(["Layer 0 sync losses", "Layer 2 sync losses"] + (["Cross-entropy losses"] if model_type=="Proxy" else []) + ["Total loss"])
    plt.title(f"{model_type} model training losses")
    plt.show()

    plt.plot(list(range(len(training_accuracies))), training_accuracies, "b")
    plt.plot(list(range(len(testing_accuracies))), testing_accuracies, "r")
    plt.xlabel("Train iteration no.")
    plt.title(f"{model_type} model accuracies throughout training")
    plt.legend(["Train accuracies", "Test accuracies"])
    plt.show()

def clear_losses():
    """
    Reset recordings of last training instance's losses and accuracies (so that the next test training metrics can be
    correctly displayed)
    """
    global layer0_sync_losses, layer2_sync_losses, crossentropy_losses, training_accuracies, testing_accuracies
    layer0_sync_losses.clear()
    layer2_sync_losses.clear()
    training_accuracies.clear()
    testing_accuracies.clear()
    crossentropy_losses.clear()

class SynchronizationLoss(nn.Module):
    """
    MSE loss between activations of project target model and model using this loss, this is to synchronize between
    the mechanisms of the models so that the convolution of the DisplacementNet will be meaningful
    """
    def __init__(self, attack_model, target_model, type):
        """
        :param attack_model: a shadow or proxy model
        :param target_model:
        """
        super(SynchronizationLoss, self).__init__()
        self.attack_model = attack_model
        self.target_model = target_model
        self.loss = nn.MSELoss()
        self.type = type


    def forward(self, y_pred, y_true, features):
        """
        :param y_pred: classification confidences of model
        :param y_true: true classifications of given data
        :param features: data to classify to compare between model's attempt at classification vs. the true
        classifications.

        y_pred and y_true go unused, only there in order to fit the abstract structure
        Computes the loss described above
        """
        # Compute attack and target model per-layer pre-activations
        # TODO: optimize by extracting these pre-activations from original model forward-prop instead of recomputing them
        attack_model_results, target_model_results = [], []
        attack_model_results.append(self.attack_model.layers[1](self.attack_model.layers[0](features)))
        attack_model_results.append(self.attack_model.layers[3](self.attack_model.layers[2](attack_model_results[0])))

        with torch.no_grad(): # To prevent training from propagating back to the target model, which must be left untouched
            target_model_results.append(self.target_model.layers[1](self.target_model.layers[0](features)))
            target_model_results.append(self.target_model.layers[3](self.target_model.layers[2](target_model_results[0])))
        if type == "SHADOW":
            layer0_weight = LAYER0_SYNC_LOSS_WEIGHT_SHADOW
            layer2_weight = LAYER2_SYNC_LOSS_WEIGHT_SHADOW
        else:
            layer0_weight = LAYER0_SYNC_LOSS_WEIGHT_PROXY
            layer2_weight = LAYER2_SYNC_LOSS_WEIGHT_PROXY

        # MSE loss on each layer of MLP
        layer0_sync_loss = layer0_weight*self.loss(attack_model_results[0], target_model_results[0])
        layer2_sync_loss = layer2_weight*self.loss(attack_model_results[1], target_model_results[1])

        global layer0_sync_losses, layer2_sync_losses, training_accuracies, testing_accuracies, test_features, test_labels
        layer0_sync_losses.append(layer0_sync_loss.item())
        layer2_sync_losses.append(layer2_sync_loss.item())
        training_accuracies.append(accuracy(predictions=y_pred[:, 1], labels=y_true))
        testing_accuracies.append(accuracy(predictions=self.attack_model(test_features)[:, 1], labels=test_labels))

        return layer0_sync_loss + layer2_sync_loss


class EntropyAndSyncLoss(nn.Module):
    """
    Weighted sum of cross entropy and Synchronization loss to obtain the optimal balance
    """
    def __init__(self, attack_model, target_model, type):
        """
        :param attack_model: a shadow or proxy model
        :param target_model:
        """
        super(EntropyAndSyncLoss, self).__init__()
        self.sync_loss = SynchronizationLoss(attack_model=attack_model, target_model=target_model,type=type)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true, features):
        """
        :param y_pred: classification confidences of model
        :param y_true: true classifications of given data
        :param features: data to classify to compare between model's attempt at classification vs. the true
        classifications
        Computes the loss described above
        """
        loss = self.loss(y_pred, y_true)
        global crossentropy_losses
        crossentropy_losses.append(loss.item())
        return loss + self.sync_loss(y_pred=y_pred, y_true=y_true, features=features)
