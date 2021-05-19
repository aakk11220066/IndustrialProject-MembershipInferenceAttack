import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from Models import MLP
from Configuration import SEEDS, NUM_EPOCHS, TARGET_TRAIN_DATA_SIZE
from GermanDataset import german_dataset
from BayesAttack import BayesAttackModel
from GeneralAttack import GeneralAttackModel
from Trainer import get_regular_model_trainer

def split_dataset(dataset):
    return dataset[:TARGET_TRAIN_DATA_SIZE], dataset[TARGET_TRAIN_DATA_SIZE:]


def get_attack_model(attack_model_class, target_model, attack_train_features, attack_train_labels):
    return attack_model_class(
        target_model=target_model,
        attack_train_features=attack_train_features,
        attack_train_labels=attack_train_labels
    )


def experiment(seed: int, attack_model_class):
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
        attack_train_labels=proxy_train_labels
    )

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
    #result = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
    # print(result)

    print(
        f"Experiment report: \n ",
        classification_report(y_true, y_pred,
            labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn'
        )
    )
    return y_true, y_pred



def main():

    experiment_results = ([experiment(seed, GeneralAttackModel) for seed in SEEDS])
    true_class,predicted_class = [],[]
    for experiment_result in experiment_results:
        true_class+=experiment_result[0]
        predicted_class+=experiment_result[1]
    print(f"Classification report: \n {classification_report(true_class, predicted_class, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')}")


if __name__ == "__main__":
    main()

#DELETE THIS COMMENT
