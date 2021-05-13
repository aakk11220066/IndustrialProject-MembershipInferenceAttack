import torch.nn as nn
import torch
from torch import Tensor
from tqdm import tqdm
from Models import LinearModel
from Configuration import  NUM_EPOCHS, SHADOW_TRAIN_DATA_SIZE, NUM_SHADOW_MODELS, MINIBATCH_SIZE, DECAY_RATE, \
    NUM_PATIENCE_EPOCHS, VERBOSE_LINEAR_TRAINING
from SyntheticDataset import _shuffle_rows


def get_linear_trainer(model: nn.Module):
    # FIXME: use paper's momentum instead of mine
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01,betas=(0.9, 0.999), eps=1e-08,amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)
    loss = nn.CrossEntropyLoss()
    return ModelTrainer(model=model, loss_fn=loss, optimization_scheduler=scheduler)


def get_conv_trainer(model: nn.Module):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01,betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)
    loss = nn.BCELoss()
    return ConvModelTrainer(model=model, loss_fn=loss, optimization_scheduler=scheduler)

class ModelTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimization_scheduler: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.scheduler = optimization_scheduler

    def train_batch(self, train_features: torch.Tensor, train_labels: torch.Tensor):
        confidence_levels = self.model(train_features)  # (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
        self.scheduler.optimizer.zero_grad()
        self.loss_fn(confidence_levels, train_labels).backward()
        self.scheduler.optimizer.step()
        self.scheduler.step()

    def accuracy(self, test_features: torch.Tensor, test_labels: torch.Tensor):
        confidence_levels = self.model(test_features)  # (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
        #class_label_predictions = confidence_levels.argmax(dim=1)  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
        class_label_predictions = torch.distributions.Categorical(probs=confidence_levels).sample()  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
        num_correct = (class_label_predictions == test_labels).sum().item()
        return num_correct / test_labels.shape[0]

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor, num_epochs: int):
        if (VERBOSE_LINEAR_TRAINING):
            print("Beginning training linear model.")
        for _ in tqdm(range(num_epochs), disable=not VERBOSE_LINEAR_TRAINING):
            self.train_batch(train_features, train_labels)
        if (VERBOSE_LINEAR_TRAINING):
            print(f"Done.  Training accuracy: {self.accuracy(train_features, train_labels)}")


def trained_linear_model(features: Tensor, class_labels: Tensor):
    model = LinearModel()
    get_linear_trainer(model).fit(features, class_labels, num_epochs=NUM_EPOCHS)
    return model


def split_dataset(features, labels):
    dataset = torch.cat([features, labels.unsqueeze(dim=-1)], dim=1)
    dataset = _shuffle_rows(dataset)
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    return features[:SHADOW_TRAIN_DATA_SIZE], features[SHADOW_TRAIN_DATA_SIZE:], \
           labels[:SHADOW_TRAIN_DATA_SIZE], labels[SHADOW_TRAIN_DATA_SIZE:]


class ConvModelTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimization_scheduler: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.scheduler = optimization_scheduler

    def training_minibatch(self, train_features, train_labels, seed=0):
        """
        A convolution model makes a model.  When training it we therefore have to generate a model and then apply it on
        the attack_train_features to attempt to make us learn to construct the optimal model in the convolutional stage
        """
        torch.random.manual_seed(seed)

        # split dataset
        shadow_features, proxy_features, shadow_labels, proxy_labels = split_dataset(train_features, train_labels)
        shadow_labels, proxy_labels = shadow_labels.long(), proxy_labels.long()

        # construct models
        shadow_model = trained_linear_model(features=shadow_features, class_labels=shadow_labels)
        proxy_model = trained_linear_model(features=proxy_features, class_labels=proxy_labels)

        membership_labels = torch.cat([
            torch.ones(shadow_features.shape[0]),
            torch.zeros(proxy_features.shape[0])
        ], dim=0)
        with torch.no_grad():
            weights = torch.stack([shadow_model.layers[0].weight, proxy_model.layers[0].weight], dim=2)
            biases = torch.stack([shadow_model.layers[0].bias, proxy_model.layers[0].bias], dim=1)

        return weights, biases, \
               torch.cat([shadow_features, proxy_features], dim=0), torch.cat([shadow_labels, proxy_labels], dim=0), \
               membership_labels



    def get_training_batch(self, train_features, train_labels, num_dataset_splits):
        print("Creating training batch")
        split_seeds = torch.arange(num_dataset_splits)

        weights, biases, x, y, membership_labels = \
            zip(*(
                self.training_minibatch(train_features, train_labels, seed=split_seeds[i])
                for i in tqdm(range(num_dataset_splits))
            ))
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

            shadow_weights = weights[:, :, :, 0]
            shadow_biases = biases[:, :, 0]
            proxy_weights = weights[:, :, :, 1]
            proxy_biases = biases[:, :, 1]

            attack_weights, attack_biases = self.model(shadow_weights, proxy_weights, shadow_biases, proxy_biases)

            membership_unnormalized_predictions = \
                attack_weights.bmm(train_features.unsqueeze(dim=-1)).squeeze(dim=-1) + attack_biases
            membership_predictions = torch.sigmoid(membership_unnormalized_predictions)
            membership_predictions = membership_predictions.gather(dim=1, index=train_labels.unsqueeze(dim=0)).squeeze()


            self.scheduler.optimizer.zero_grad()
            loss = self.loss_fn(membership_predictions, membership_labels)
            loss.backward()
            self.scheduler.optimizer.step()
            self.scheduler.step()

            #if progress_count % 50 == 0:
            #    print(f"Finished training DisplacementNet on datapoint {progress_count}/{len(training_dl)}")
        return loss.item()

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor,
            num_epochs=NUM_EPOCHS, num_shadow_models=NUM_SHADOW_MODELS):
        """
        num_shadow_models is N from paper
        """
        print("-----Beginning training conv displacement model.-----")
        training_dl = self.get_training_batch(train_features, train_labels, num_shadow_models)

        print("Training conv model on proxy-shadow model pairs")
        best_loss = float('inf')
        num_epochs_without_improvement = 0
        for _ in tqdm(range(num_epochs)):
            loss = self.train_epoch(training_dl)

            if loss >= best_loss:
                num_epochs_without_improvement += 1
            else:
                num_epochs_without_improvement = 0
                best_loss = loss
            if num_epochs_without_improvement > NUM_PATIENCE_EPOCHS:
                print(f"Stopping early due to no improvement for {NUM_PATIENCE_EPOCHS} epochs")
                break

        print(f"Done.")

