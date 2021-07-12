import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from Models import MLP
from Configuration import SEEDS, NUM_EPOCHS, TARGET_TRAIN_DATA_SIZE
from GermanDataset import german_dataset
from BayesAttack import BayesAttackModel
from GeneralAttack import GeneralAttackModel
from Trainer import get_regular_model_trainer
from Models import DisplacementNet
from Trainer import layer0_weight, layer2_weight, layer0_bias, layer2_bias
def split_dataset(dataset):
    """
    :param dataset:
    :return splitted data to training set and test set
    """
    return dataset[:TARGET_TRAIN_DATA_SIZE], dataset[TARGET_TRAIN_DATA_SIZE:]



def get_attack_model(attack_model_class, target_model, attack_train_features, attack_train_labels,
                     test_features, test_labels):
    """
    :param attack_model_class: class of attack model
    :param target_model: The attacked model
    :param attack_train_features: attack model training data
    :param attack_train_labels: attack model training data
    :param test_features: attack model test data
    :param test_labels: attack model test data
    :return trained attack model
    """
    return attack_model_class(
        target_model=target_model,
        attack_train_features=attack_train_features,
        attack_train_labels=attack_train_labels,
        test_features=test_features, test_labels=test_labels
    )


def experiment(seed: int, attack_model_class):
    """
    :param seed:
    :param attack_model_class: class of attack model
    :return y_pred: predicted values, y_true: true labels
    This function making an experiment. At first, build and fit a MLP target model(a model we want to attack). Then, build anf fit a
    MLP attack model. In the end, testing our attack model.
    """
    torch.random.manual_seed(seed)
    train_features, train_labels, test_features, test_labels = german_dataset(filename="German.csv")
    train_features = nn.functional.normalize(train_features, dim=0)
    test_features = nn.functional.normalize(test_features, dim=0)

    target_train_features, proxy_train_features = split_dataset(train_features)
    target_train_labels, proxy_train_labels = split_dataset(train_labels)

    target_model = MLP()
    trainer = get_regular_model_trainer(model=target_model)
    trainer.fit(target_train_features, target_train_labels, num_epochs=NUM_EPOCHS)

    print(f"Target model test accuracy = {trainer.accuracy(test_features=test_features, test_labels=test_labels)}")


    attack_model = get_attack_model(
        attack_model_class=attack_model_class,
        target_model=target_model,
        attack_train_features=proxy_train_features,
        attack_train_labels=proxy_train_labels,
        test_features=test_features, test_labels=test_labels
    )
    best_model = torch.load("MyModel") # load the best model we saved during the attack model training
    best_model.eval()


    attack_model.weights_displacer.layer0_conv = best_model.layer0_conv
    attack_model.weights_displacer.layer2_conv = best_model.layer2_conv

    correct_intrainset_predictions = (
        attack_model(
            x=target_train_features[:test_features.shape[0]],
            y=target_train_labels[:test_features.shape[0]]
        )
    ).sum().item()
    correct_outtrainset_predictions = attack_model(x=test_features, y=test_labels).logical_not().sum().item()
    accuracy = (correct_intrainset_predictions + correct_outtrainset_predictions) / \
               (2 * test_features.shape[0])
    print(f"True/false positives accuracy: {correct_intrainset_predictions / test_features.shape[0]}, "
          f"true/false negatives accuracy: {correct_outtrainset_predictions / test_features.shape[0]}, "
          f"total accuracy: {accuracy}\n"
          )
    y_true = [False] * test_features.shape[0] + [True] * test_features.shape[0]
    y_pred = list(attack_model(x=test_features, y=test_labels)) + \
             list(attack_model(
                 x=target_train_features[:test_features.shape[0]],
                 y=target_train_labels[:test_features.shape[0]]
             )
    )

    print(
        f"Experiment report: \n ",
        classification_report(y_true, y_pred,
            labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'
        )
    )
    return y_true, y_pred



def main():
    """
    Making few experiments. Printing classification_report of all experiments.
    """
    experiment_results = ([experiment(seed, GeneralAttackModel) for seed in SEEDS])
    true_class,predicted_class = [],[]
    for experiment_result in experiment_results:
        true_class+=experiment_result[0]
        predicted_class+=experiment_result[1]
    print(f"Classification report: \n {classification_report(true_class, predicted_class, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')}")


if __name__ == "__main__":
    main()
#
