import torch
import pandas as pd
from Configuration import BATCH_SIZE, NUM_CLASS_FEATURES, NUM_CLASSES, TRAIN_DATA_SIZE


def _shuffle_rows(x):
    return x[torch.randperm(x.shape[0])]


def german_dataset(filename = "German.csv"):
    """
    returns random dataset of training shape (batch_size, #x) and label shape (batch_size,)
    split into (attack_train_features, attack_train_labels), (test_features, test_labels)
    """
    dataset = torch.Tensor(pd.read_csv(filename).values)
    dataset = _shuffle_rows(dataset) # NOTE: unnecessary because not using mini-batches, but no harm done
    dataset_features, dataset_labels = dataset[:, 1:], dataset[:, 0].long()
    assert(NUM_CLASS_FEATURES == dataset_features.shape[1])
    assert(NUM_CLASSES == dataset_labels.unique().shape[0])
    assert(BATCH_SIZE == dataset_features.shape[0] and BATCH_SIZE == dataset_labels.shape[0])

    # split data into train/test
    train_data_size = TRAIN_DATA_SIZE
    test_data_size = dataset.shape[0] - train_data_size
    return dataset_features[:train_data_size], dataset_labels[:train_data_size], \
           dataset_features[-test_data_size:], dataset_labels[-test_data_size:]

#DELETE THIS COMMENT
