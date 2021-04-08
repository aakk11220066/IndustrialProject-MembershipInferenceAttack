import torch.nn as nn
import torch
from tqdm import tqdm


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
        class_label_predictions = confidence_levels.argmax(dim=1)  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
        num_correct = (class_label_predictions == test_labels).sum().item()
        return num_correct / test_labels.shape[0]

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor, num_epochs: int):
        for _ in tqdm(range(num_epochs)):
            self.train_batch(train_features, train_labels)
        print(f"Done.  Training accuracy: {self.accuracy(train_features, train_labels)}")
