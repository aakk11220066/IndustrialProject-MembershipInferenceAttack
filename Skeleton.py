import torch
from sklearn.metrics import classification_report
from statistics import mean
from Models import LinearModel
from Configuration import SEEDS, NUM_EPOCHS, get_trainer, TARGET_TRAIN_DATA_SIZE
from SyntheticDataset import synthetic_dataset
from BayesAttack import BayesAttackModel


def split_dataset(dataset):
    return dataset[:TARGET_TRAIN_DATA_SIZE], dataset[TARGET_TRAIN_DATA_SIZE:]


def experiment(seed: int):
    torch.random.manual_seed(seed)
    train_features, train_labels, test_features, test_labels = synthetic_dataset()
    target_train_features, proxy_train_features = split_dataset(train_features)
    target_train_labels, proxy_train_labels = split_dataset(train_labels)

    target_model = LinearModel()
    trainer = get_trainer(model=target_model)
    trainer.fit(target_train_features, target_train_labels, num_epochs=NUM_EPOCHS)

    attack_model = BayesAttackModel(
        target_model=target_model,
        proxy_train_features=proxy_train_features,
        proxy_train_labels=proxy_train_labels
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
          f"total accuracy: {accuracy}"
          )
    y_true = [False] * 160 + [True]*160
    y_pred = list(attack_model(x=test_features, y=test_labels)) + list(attack_model(
            x=target_train_features[:test_features.shape[0]],
            y=target_train_labels[:test_features.shape[0]]))
    #result = classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
    # print(result)

    return y_true, y_pred



def main():

    my_list = ([experiment(seed) for seed in SEEDS])
    y_true,y_pred = [],[]
    for elem in my_list:
        y_true+=elem[0]
        y_pred+=elem[1]
    print(f"Classification report: \n {classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')}")


if __name__ == "__main__":
    main()