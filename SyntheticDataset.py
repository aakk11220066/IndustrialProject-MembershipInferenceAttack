import torch
from Configuration import BATCH_SIZE, NUM_CLASS_FEATURES, NUM_CLASSES, FEATURE_STD_RANGE, TRAIN_DATA_SIZE


def _shuffle_rows(x):
    return x[torch.randperm(x.shape[0])]


def synthetic_dataset():
    """
    returns random dataset of training shape (batch_size, #x) and label shape (batch_size,)
    split into (attack_train_features, attack_train_labels), (test_features, test_labels) distributed with 0-mean
    Gaussian distribution with STD as set in configuration
    """
    # generate data x.  Shape: (BATCH_SIZE // NUM_CLASSES, NUM_CLASSES, NUM_CLASS_FEATURES)
    feature_std_matrix = torch.diag(torch.distributions.Uniform(*FEATURE_STD_RANGE).sample((NUM_CLASS_FEATURES,)))
    class_means = torch.rand(NUM_CLASSES).unsqueeze(dim=0).unsqueeze(dim=2)
    class_features = torch.matmul(
        feature_std_matrix,
        torch.randn(size=(BATCH_SIZE // NUM_CLASSES, NUM_CLASSES, NUM_CLASS_FEATURES)).transpose(1, 2) # Gaussian
    ).transpose(1, 2) + class_means

    # label data.  Features shape: (BATCH_SIZE, NUM_CLASSES, NUM_CLASS_FEATURES).  Labels shape: (BATCH_SIZE)
    labels = torch.arange(NUM_CLASSES) \
        .unsqueeze(dim=1).unsqueeze(dim=0).expand(BATCH_SIZE // NUM_CLASSES, NUM_CLASSES, 1)
    dataset = torch.cat((class_features, labels), dim=2)
    dataset = dataset.flatten(end_dim=1)
    dataset = _shuffle_rows(dataset) # NOTE: unnecessary because not using mini-batches, but no harm done
    dataset_features, dataset_labels = dataset[:, :-1], dataset[:, -1].long()

    # split data into train/test
    train_data_size = TRAIN_DATA_SIZE
    test_data_size = dataset.shape[0] - train_data_size
    return dataset_features[:train_data_size], dataset_labels[:train_data_size], \
           dataset_features[-test_data_size:], dataset_labels[-test_data_size:]