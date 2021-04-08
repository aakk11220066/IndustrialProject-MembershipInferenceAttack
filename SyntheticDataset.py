import torch
from Configuration import BATCH_SIZE, NUM_CLASS_FEATURES, NUM_CLASSES, FEATURE_STD_RANGE, TRAIN_TEST_RATIO


def shuffle_rows(x):
    return x[torch.randperm(x.shape[0])]


def synthetic_dataset():
    """
    returns random dataset of shape (batch_size, #features + 1)
    split into (train_features, train_labels), (test_features, test_labels)
    """
    # generate data features.  Shape: (BATCH_SIZE // NUM_CLASSES, NUM_CLASSES, NUM_CLASS_FEATURES)
    feature_std_matrix = torch.diag(torch.distributions.Uniform(*FEATURE_STD_RANGE).sample((NUM_CLASS_FEATURES,)))
    class_means = torch.rand(NUM_CLASSES).unsqueeze(dim=0).unsqueeze(dim=2)
    class_features = torch.matmul(
        feature_std_matrix,
        torch.randn(size=(BATCH_SIZE // NUM_CLASSES, NUM_CLASSES, NUM_CLASS_FEATURES)).transpose(1, 2)
    ).transpose(1, 2) + class_means

    # label data.  Features shape: (BATCH_SIZE, NUM_CLASS_FEATURES).  Labels shape: (BATCH_SIZE)
    labels = torch.arange(NUM_CLASSES).unsqueeze(dim=1).unsqueeze(dim=0).expand(BATCH_SIZE // NUM_CLASSES, NUM_CLASSES,
                                                                                1)
    dataset = torch.cat((class_features, labels), dim=2)
    dataset = dataset.flatten(end_dim=1)
    dataset = shuffle_rows(dataset)
    dataset_features, dataset_labels = dataset[:, :-1], dataset[:, -1].long()

    # split data into train/test
    train_data_size = int(dataset.shape[0] * TRAIN_TEST_RATIO)
    test_data_size = dataset.shape[0] - train_data_size
    return dataset_features[:train_data_size], dataset_labels[:train_data_size], \
           dataset_features[-test_data_size:], dataset_labels[-test_data_size:]
