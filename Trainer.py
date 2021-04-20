import torch.nn as nn
import torch
from torch import Tensor
from tqdm import tqdm
from Models import LinearModel
from Configuration import get_linear_trainer, NUM_EPOCHS, SHADOW_TRAIN_DATA_SIZE


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


def split_dataset(dataset): # FIXME: should shuffle dataset
    return dataset[:SHADOW_TRAIN_DATA_SIZE], dataset[SHADOW_TRAIN_DATA_SIZE:]


class ConvModelTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_batch(self, train_features, train_labels):
        """
        A convolution model makes a model.  When training it we therefore have to generate a model and then apply it on
        the train_features to attempt to make us learn to construct the optimal model in the convolutional stage
        """
        # split dataset
        shadow_features, proxy_features = split_dataset(train_features[:, :-1])
        shadow_labels, proxy_labels = split_dataset(train_features[:, -1])

        # construct models
        shadow_model = trained_linear_model(features=shadow_features, class_labels=shadow_labels)
        proxy_model = trained_linear_model(features=proxy_features, class_labels=proxy_labels)

        self.model(shadow_model, proxy_model)(train_features).round()
        # FIXME: train against labels
        # FIXME: this training should be on entire dataset of N pairs of models, not on each pair alone

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor, num_dataset_splits: int):
        """
        num_dataset_splits is N from paper
        """
        split_seeds = torch.arange(num_dataset_splits)

        print("Beginning training conv displacement model.")
        for i in tqdm(range(num_dataset_splits)):
            torch.random.manual_seed(split_seeds[i])
            self.train_batch(train_features, train_labels)
        print(f"Done.")