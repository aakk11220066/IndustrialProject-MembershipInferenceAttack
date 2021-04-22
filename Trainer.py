import torch.nn as nn
import torch
from torch import Tensor
from tqdm import tqdm
from Models import LinearModel
from Configuration import get_linear_trainer, NUM_EPOCHS, SHADOW_TRAIN_DATA_SIZE
from SyntheticDataset import _shuffle_rows


class ModelTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_batch(self, train_features: torch.Tensor, train_labels: torch.Tensor):
        confidence_levels = self.model(train_features)  # (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
        self.optimizer.zero_grad()
        self.loss_fn(confidence_levels, train_labels).backward()
        self.optimizer.step()

    def accuracy(self, test_features: torch.Tensor, test_labels: torch.Tensor):
        confidence_levels = self.model(test_features)  # (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
        #class_label_predictions = confidence_levels.argmax(dim=1)  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
        class_label_predictions = torch.distributions.Categorical(probs=confidence_levels).sample()  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
        num_correct = (class_label_predictions == test_labels).sum().item()
        return num_correct / test_labels.shape[0]

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor, num_epochs: int):
        print("Beginning training linear model.")
        for _ in tqdm(range(num_epochs)):
            self.train_batch(train_features, train_labels)
        print(f"Done.  Training accuracy: {self.accuracy(train_features, train_labels)}")


def trained_linear_model(features: Tensor, class_labels: Tensor):
    model = LinearModel()
    get_linear_trainer(model).fit(features, class_labels, num_epochs=NUM_EPOCHS)
    return model


def split_dataset(dataset):
    _shuffle_rows(dataset)
    return dataset[:SHADOW_TRAIN_DATA_SIZE], dataset[SHADOW_TRAIN_DATA_SIZE:]


class ConvModelTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def training_minibatch(self, train_features, train_labels, seed=0):
        """
        A convolution model makes a model.  When training it we therefore have to generate a model and then apply it on
        the train_features to attempt to make us learn to construct the optimal model in the convolutional stage
        """
        torch.random.manual_seed(seed)

        # split dataset
        shadow_features, proxy_features = split_dataset(train_features)
        shadow_labels, proxy_labels = split_dataset(train_labels)

        # construct models
        shadow_model = trained_linear_model(features=shadow_features, class_labels=shadow_labels)
        proxy_model = trained_linear_model(features=proxy_features, class_labels=proxy_labels)

        membership_labels = torch.cat([
            torch.ones(shadow_features.shape[0]),
            torch.zeros(proxy_features.shape[0])
        ], dim=1)

        weights = torch.stack([shadow_model.layers[0].weight, proxy_model.layers[0].weight], dim=2)
        biases = torch.stack([shadow_model.layers[0].bias, proxy_model.layers[0].bias], dim=2)
        return weights, biases, torch.cat([shadow_features, proxy_features], dim=0), membership_labels


    def get_training_batch(self, train_features, train_labels, num_dataset_splits):
        print("Creating training batch")
        split_seeds = torch.arange(num_dataset_splits)

        weights, biases, x, membership_labels =\
            zip(*(self.training_minibatch(train_features, train_labels, seed=split_seeds[i]) for i in range(num_dataset_splits)))
        weights = weights.expand(x.shape[0], *weights.shape)
        biases = biases.expand(x.shape[0], *biases.shape)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorsDataset(weights, biases, x, membership_labels),
            shuffle=True
        )

    def train_epoch(self, training_dl: torch.utils.data.DataLoader):
        for membership_example in training_dl:
            weights, biases, train_features, membership_label = membership_example
            shadow_model = LinearModel()
            shadow_model.layers[0].weights = weights[0, :, :]
            shadow_model.layers[0].bias = biases[0, :]
            proxy_model = LinearModel()
            proxy_model.layers[0].weights = weights[1, :, :]
            proxy_model.layers[0].bias = biases[1, :]

            membership_prediction = self.model(shadow_model, proxy_model)(train_features).round()
            self.optimizer.zero_grad()
            self.loss_fn(membership_prediction, membership_label).backward()
            self.optimizer.step()

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor, num_dataset_splits: int):
        """
        num_dataset_splits is N from paper
        """
        print("Beginning training conv displacement model.")
        training_dl = self.get_train_batch(train_features, train_labels, num_dataset_splits)
        for _ in tqdm(range(NUM_EPOCHS)):
            self.train_epoch(training_dl)
        print(f"Done.")