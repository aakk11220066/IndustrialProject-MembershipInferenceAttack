import torch.nn as nn
import torch
from torch import Tensor
from tqdm import tqdm
from Models import LinearModel
from Configuration import  NUM_EPOCHS, SHADOW_TRAIN_DATA_SIZE, NUM_SHADOW_MODELS, MINIBATCH_SIZE
from SyntheticDataset import _shuffle_rows


def get_linear_trainer(model: nn.Module):
    # FIXME: use paper's momentum instead of mine
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.01, nesterov=True)
    loss = nn.CrossEntropyLoss()
    return ModelTrainer(model=model, loss_fn=loss, optimizer=optimizer)

def get_conv_trainer(model: nn.Module):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.01, nesterov=True)
    loss = nn.L1Loss()
    return ConvModelTrainer(model=model, loss_fn=loss, optimizer=optimizer)

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
        the attack_train_features to attempt to make us learn to construct the optimal model in the convolutional stage
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
        ], dim=0)

        weights = torch.stack([shadow_model.layers[0].weight, proxy_model.layers[0].weight], dim=2)
        biases = torch.stack([shadow_model.layers[0].bias, proxy_model.layers[0].bias], dim=1)
        return weights, biases, torch.cat([shadow_features, proxy_features], dim=0), train_labels, membership_labels


    def get_training_batch(self, train_features, train_labels, num_dataset_splits):
        print("Creating training batch")
        split_seeds = torch.arange(num_dataset_splits)

        weights, biases, x, y, membership_labels = \
            zip(*(self.training_minibatch(train_features, train_labels, seed=split_seeds[i]) for i in range(num_dataset_splits)))
        weights, biases, x, y, membership_labels = \
            map(lambda data_tuple: torch.stack(data_tuple, dim=0), (weights, biases, x, y, membership_labels))
        weights = weights.unsqueeze(dim=1).expand(weights.shape[0], x.shape[1], *weights.shape[1:])
        biases = biases.unsqueeze(dim=1).expand(biases.shape[0], x.shape[1], *biases.shape[1:])
        '''
        Definitions: 
        matrix_pair = (NUM_CLASSES, NUM_CLASS_FEATURES, |{proxy_dataset, shadow_dataset}|)
        bias_pair = (NUM_CLASSES, |{proxy_dataset, shadow_dataset}|)

        weights.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE, matrix_pair)
        biases.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE, bias_pair)
        x.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE, NUM_CLASS_FEATURES)
        y.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE)
        membership_labels.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE)
        '''
        weights, biases, x, y, membership_labels = \
            map(lambda tensor: tensor.flatten(end_dim=1), (weights, biases, x, y, membership_labels))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(weights, biases, x, y, membership_labels),
            shuffle=True,
            batch_size = MINIBATCH_SIZE
        )

    def train_epoch(self, training_dl: torch.utils.data.DataLoader):
        for progress_count, membership_example in enumerate(training_dl):
            weights, biases, train_features, train_labels, membership_labels = membership_example

            shadow_weights = weights[:, :, :, 1]
            shadow_biases = biases[:, :, 1]
            proxy_weights = weights[:, :, :, 0]
            proxy_biases = biases[:, :, 0]

            attack_weights, attack_biases = self.model(shadow_weights, proxy_weights, shadow_biases, proxy_biases)

            membership_predictions = torch.sigmoid(attack_weights.bmm(train_features.unsqueeze(dim=-1)).squeeze(dim=-1) + attack_biases)
            membership_predictions = membership_predictions.gather(dim=1, index=train_labels.unsqueeze(dim=0)).round().squeeze()

            self.optimizer.zero_grad()
            loss = self.loss_fn(membership_predictions, membership_labels)
            loss.backward()
            self.optimizer.step()

            if progress_count % 50 == 0:
                print(f"Finished training DisplacementNet on datapoint {progress_count}/{len(training_dl)}")

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor,
            num_epochs=NUM_EPOCHS, num_shadow_models=NUM_SHADOW_MODELS):
        """
        num_shadow_models is N from paper
        """
        print("Beginning training conv displacement model.")
        training_dl = self.get_training_batch(train_features, train_labels, num_shadow_models)
        for _ in tqdm(range(num_epochs)):
            self.train_epoch(training_dl)
        print(f"Done.")

#DELETE THIS COMMENT
