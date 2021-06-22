import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch import Tensor
from tqdm import tqdm
from Models import LinearModel, MLP
from Configuration import  NUM_EPOCHS, SHADOW_TRAIN_DATA_SIZE, PROXY_TRAIN_DATA_SIZE, NUM_SHADOW_MODELS, MINIBATCH_SIZE, DECAY_RATE, \
    NUM_PATIENCE_EPOCHS, VERBOSE_REGULAR_TRAINING, VERBOSE_CONVOLUTION_TRAINING, SHOW_SHADOW_PROXY_LOSS_GRAPHS
from SyntheticDataset import _shuffle_rows
from Loss import EntropyAndSyncLoss, SynchronizationLoss
from Loss import display_losses, clear_losses, init_test_data

layer0_weight, layer2_weight, layer0_bias, layer2_bias= 0,0,0,0
def get_regular_model_trainer(model: nn.Module, loss=nn.CrossEntropyLoss()):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01,betas=(0.9, 0.999), eps=1e-08,amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)
    result = ModelTrainer(model=model, loss_fn=loss, optimization_scheduler=scheduler)
    if type(loss) == nn.CrossEntropyLoss:
        return result
    return AttackModelTrainer(model=model, loss_fn=loss, optimization_scheduler=scheduler)


def get_conv_trainer(model: nn.Module, target_model: nn.Module, test_features, test_labels):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.01 ,betas=(0.85, 0.999), eps=1e-05, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)
    loss = nn.BCELoss()
    return ConvModelTrainer(model=model, loss_fn=loss, optimization_scheduler=scheduler, target_model=target_model,
                            test_features=test_features, test_labels=test_labels)

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
        if (VERBOSE_REGULAR_TRAINING):
            print("Beginning training regular model.")
        for _ in tqdm(range(num_epochs), disable=not VERBOSE_REGULAR_TRAINING):
            self.train_batch(train_features, train_labels)
        if (VERBOSE_REGULAR_TRAINING):
            print(f"Done.  Training accuracy: {self.accuracy(train_features, train_labels)}")


class AttackModelTrainer(ModelTrainer):
    def train_batch(self, train_features: torch.Tensor, train_labels: torch.Tensor):
        confidence_levels = self.model(train_features)  # (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
        self.scheduler.optimizer.zero_grad()
        self.loss_fn(confidence_levels, train_labels, train_features).backward()
        self.scheduler.optimizer.step()
        self.scheduler.step()


def trained_linear_model(features: Tensor, class_labels: Tensor):
    model = LinearModel()
    get_regular_model_trainer(model).fit(features, class_labels, num_epochs=NUM_EPOCHS)
    return model

def trained_proxy_MLP_model(features: Tensor, class_labels: Tensor, target_model: nn.Module, test_features, test_labels):
    model = MLP()
    trainer = get_regular_model_trainer(model, loss=EntropyAndSyncLoss(model, target_model, type="PROXY"))
    trainer.fit(features, class_labels, num_epochs=NUM_EPOCHS)
    if SHOW_SHADOW_PROXY_LOSS_GRAPHS:
        display_losses(model_type="Proxy")
    clear_losses()
    #print(f"Proxy model test accuracy = {trainer.accuracy(test_features=test_features, test_labels=test_labels)}")
    return model

def trained_shadow_MLP_model(features: Tensor, class_labels: Tensor, target_model: nn.Module, test_features, test_labels):
    model = MLP()
    trainer = get_regular_model_trainer(model, loss=EntropyAndSyncLoss(model, target_model, type="SHADOW"))
    class_labels = target_model(features).argmax(dim=1)
    trainer.fit(features, class_labels, num_epochs=NUM_EPOCHS)
    if SHOW_SHADOW_PROXY_LOSS_GRAPHS:
        display_losses(model_type="Shadow")
    clear_losses()
    #print(f"\nShadow model test accuracy = {trainer.accuracy(test_features=test_features, test_labels=test_labels)}")
    return model


def split_dataset(features, labels):
    dataset = torch.cat([features, labels.unsqueeze(dim=-1)], dim=1)
    dataset = _shuffle_rows(dataset)
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    return features[: SHADOW_TRAIN_DATA_SIZE], \
           features[SHADOW_TRAIN_DATA_SIZE : SHADOW_TRAIN_DATA_SIZE + PROXY_TRAIN_DATA_SIZE], \
           features[SHADOW_TRAIN_DATA_SIZE + PROXY_TRAIN_DATA_SIZE :], \
           labels[: SHADOW_TRAIN_DATA_SIZE], \
           labels[SHADOW_TRAIN_DATA_SIZE : SHADOW_TRAIN_DATA_SIZE + PROXY_TRAIN_DATA_SIZE], \
           labels[SHADOW_TRAIN_DATA_SIZE + PROXY_TRAIN_DATA_SIZE :]


# DELETEME
def temp_acc(membership_predictions, membership_labels):
    confidence_levels = membership_predictions  # (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
    test_labels = membership_labels
    # class_label_predictions = confidence_levels.argmax(dim=1)  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
    # class_label_predictions = torch.stack([1 - confidence_levels, confidence_levels], dim=1).multinomial(1).squeeze(
    #      dim=1)  # (BATCH_SIZE, NUM_CLASSES) -> (BATCH_SIZE,)
    class_label_predictions = confidence_levels.round()
    num_correct = (class_label_predictions == test_labels).sum().item()
    return num_correct / test_labels.shape[0]


class ConvModelTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimization_scheduler: torch.optim.Optimizer,
                 target_model, test_features, test_labels):
        self.model = model
        self.loss_fn = loss_fn
        self.scheduler = optimization_scheduler
        self.target_model = target_model

        # DELETEME
        self.test_features = test_features
        self.test_labels = test_labels

        # DELETEME
        init_test_data(test_features, test_labels)

    def training_minibatch(self, train_features, train_labels, seed=0):
        """
        A convolution model makes a model.  When training it we therefore have to generate a model and then apply it on
        the attack_train_features to attempt to make us learn to construct the optimal model in the convolutional stage
        """
        torch.random.manual_seed(seed)

        # split dataset
        shadow_features, proxy_features, holdout_features, shadow_labels, proxy_labels, holdout_labels = \
            split_dataset(train_features, train_labels)
        shadow_labels, proxy_labels, holdout_labels = shadow_labels.long(), proxy_labels.long(), holdout_labels.long()

        # construct models
        shadow_model = trained_shadow_MLP_model(features=shadow_features, class_labels=shadow_labels, target_model=self.target_model, test_features=self.test_features, test_labels=self.test_labels)
        proxy_model = trained_proxy_MLP_model(features=proxy_features, class_labels=proxy_labels, target_model=self.target_model, test_features=self.test_features, test_labels=self.test_labels)

        membership_labels = torch.cat([
            torch.ones(shadow_features.shape[0]),
            torch.zeros(holdout_features.shape[0])
        ], dim=0)
        with torch.no_grad():
            layer0_weights = torch.stack([shadow_model.layers[0].weight, proxy_model.layers[0].weight], dim=2)
            layer0_biases = torch.stack([shadow_model.layers[0].bias, proxy_model.layers[0].bias], dim=1)
            layer2_weights = torch.stack([shadow_model.layers[2].weight, proxy_model.layers[2].weight], dim=2)
            layer2_biases = torch.stack([shadow_model.layers[2].bias, proxy_model.layers[2].bias], dim=1)

        return layer0_weights, layer0_biases, layer2_weights, layer2_biases, \
               torch.cat([shadow_features, holdout_features], dim=0), torch.cat([shadow_labels, holdout_labels], dim=0), \
               membership_labels



    def get_training_batch(self, train_features, train_labels, num_dataset_splits):
        print("Creating training batch")
        split_seeds = torch.randperm(num_dataset_splits**2)[:num_dataset_splits]

        layer0_weights, layer0_biases, layer2_weights, layer2_biases, x, y, membership_labels = \
            zip(*(
                self.training_minibatch(train_features, train_labels, seed=split_seeds[i])
                for i in tqdm(range(num_dataset_splits), disable=not VERBOSE_CONVOLUTION_TRAINING)
            ))
        layer0_weights, layer0_biases, layer2_weights, layer2_biases, x, y, membership_labels = \
            map(
                lambda data_tuple: torch.stack(data_tuple, dim=0), 
                (layer0_weights, layer0_biases, layer2_weights, layer2_biases, x, y, membership_labels)
            )
        layer0_weights = layer0_weights\
            .unsqueeze(dim=1)\
            .expand(layer0_weights.shape[0], x.shape[1], *layer0_weights.shape[1:])
        layer0_biases = layer0_biases\
            .unsqueeze(dim=1)\
            .expand(layer0_biases.shape[0], x.shape[1], *layer0_biases.shape[1:])
        layer2_weights = layer2_weights\
            .unsqueeze(dim=1)\
            .expand(layer2_weights.shape[0], x.shape[1], *layer2_weights.shape[1:])
        layer2_biases = layer2_biases\
            .unsqueeze(dim=1)\
            .expand(layer2_biases.shape[0], x.shape[1], *layer2_biases.shape[1:])
        '''
        Definitions: 
        matrix_pair = (NUM_CLASSES, NUM_CLASS_FEATURES, |{holdout_dataset, shadow_dataset}|)
        bias_pair = (NUM_CLASSES, |{holdout_dataset, shadow_dataset}|)

        weights.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE, matrix_pair)
        biases.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE, bias_pair)
        x.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE, NUM_CLASS_FEATURES)
        y.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE)
        membership_labels.shape == (NUM_SHADOW_MODELS, ATTACK_TRAIN_DATA_SIZE)
        '''
        layer0_weights, layer0_biases, layer2_weights, layer2_biases, x, y, membership_labels = \
            map(
                lambda tensor: tensor.flatten(end_dim=1), 
                (layer0_weights, layer0_biases, layer2_weights, layer2_biases, x, y, membership_labels)
            )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                layer0_weights, layer0_biases, layer2_weights, layer2_biases, x, y, membership_labels
            ),
            shuffle=True,
            batch_size = MINIBATCH_SIZE
        )

    def train_epoch(self, training_dl: torch.utils.data.DataLoader):
        losses = []
        accuracies = []
        for progress_count, membership_example in enumerate(training_dl):
            layer0_weights, layer0_biases, layer2_weights, layer2_biases, train_features, train_labels, membership_labels = \
                membership_example

            layer0_shadow_weights = layer0_weights[:, :, :, 0]
            layer0_shadow_biases = layer0_biases[:, :, 0]
            layer0_proxy_weights = layer0_weights[:, :, :, 1]
            layer0_proxy_biases = layer0_biases[:, :, 1]
            layer2_shadow_weights = layer2_weights[:, :, :, 0]
            layer2_shadow_biases = layer2_biases[:, :, 0]
            layer2_proxy_weights = layer2_weights[:, :, :, 1]
            layer2_proxy_biases = layer2_biases[:, :, 1]

            layer0_attack_weights, layer0_attack_biases, layer2_attack_weights, layer2_attack_biases = self.model(
                layer0_shadow_weights, layer0_proxy_weights, layer0_shadow_biases, layer0_proxy_biases,
                layer2_shadow_weights, layer2_proxy_weights, layer2_shadow_biases, layer2_proxy_biases
            )
            layer0_attack_weights = layer0_attack_weights.squeeze(dim=1)
            layer2_attack_weights = layer2_attack_weights.squeeze(dim=1)

            membership_unnormalized_predictions = \
                layer0_attack_weights.bmm(train_features.unsqueeze(dim=-1)).squeeze(dim=-1) + \
                layer0_attack_biases
            membership_unnormalized_predictions = torch.relu(membership_unnormalized_predictions)
            membership_unnormalized_predictions = \
                layer2_attack_weights.bmm(membership_unnormalized_predictions.unsqueeze(dim=-1)).squeeze(dim=-1) + \
                layer2_attack_biases
            membership_predictions = torch.sigmoid(membership_unnormalized_predictions)
            membership_predictions = membership_predictions.gather(dim=1, index=train_labels.unsqueeze(dim=-1)).squeeze()

            self.scheduler.optimizer.zero_grad()
            loss = self.loss_fn(membership_predictions, membership_labels)
            losses.append(loss.item())
            accuracies.append(temp_acc(membership_predictions, membership_labels))

            # DEBUG
            #self.model.layer0_conv.weight = nn.Parameter(torch.Tensor([[[[torch.normal(torch.Tensor([0.3]), torch.Tensor([0.3]))]], [[torch.normal(torch.Tensor([0.3]), torch.Tensor([0.2]))]]]]))
            #self.model.layer0_conv.bias = nn.Parameter(torch.Tensor([torch.normal(torch.Tensor([0.2]), torch.Tensor([0.1]))]))
            #self.model.layer2_conv.weight = nn.Parameter(torch.Tensor([[[[torch.normal(torch.Tensor([-0.4]), torch.Tensor([0.25]))]], [[torch.normal(torch.Tensor([0.0]), torch.Tensor([0.2]))]]]]))
            #self.model.layer2_conv.bias = nn.Parameter(torch.Tensor([torch.normal(torch.Tensor([0]), torch.Tensor([0.1]))]))
            # DEBUG

            #'''
            loss.backward()
            self.scheduler.optimizer.step()
            self.scheduler.step()
            #''' # DEBUG


            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(self.model, "MyModel")
                print(f"Good Loss in epoch = {self.best_loss} -> model saved!")

        return losses, accuracies

    def fit(self, train_features: torch.Tensor, train_labels: torch.Tensor,
            num_epochs=NUM_EPOCHS, num_shadow_models=NUM_SHADOW_MODELS):
        """
        num_shadow_models is N from paper
        """
        print("-----Beginning training conv displacement model.-----")
        training_dl = self.get_training_batch(train_features, train_labels, num_shadow_models)

        print("Training conv model on proxy-shadow model pairs")
        losses = []
        accuracies = []
        self.best_loss = float('inf')
        num_epochs_without_improvement = 0
        for i in tqdm(range(num_epochs)):
            losses_addendum, accuracies_addendum = self.train_epoch(training_dl)
            losses += losses_addendum
            accuracies += accuracies_addendum
            loss = losses[-1]

            if loss > self.best_loss:
                num_epochs_without_improvement += 1
            else:
                num_epochs_without_improvement = 0
                self.best_loss = loss
                best_acc = accuracies[-1]
                # layer0_weight, layer2_weight, layer0_bias, layer2_bias =\
                #     torch.nn.Parameter(self.model.layer0_conv.weight), self.model.layer2_conv.weight, self.model.layer0_conv.bias, self.model.layer2_conv.bias
                # conv0 = self.model.layer0_conv
                # conv2 = self.model.layer2_conv
                torch.save(self.model,"MyModel")
                print(f"Good Loss = {self.best_loss} -> model saved!")
            if num_epochs_without_improvement > NUM_PATIENCE_EPOCHS and i>8:
                print(f"\nStopping early due to no improvement for {NUM_PATIENCE_EPOCHS} epochs")
                break

        # DELETEME
        plt.plot(list(range(len(losses))), losses[:len(losses)], color="b")
        plt.xlabel("Minibatch no.")
        plt.ylabel("Attack loss")
        plt.show()
        plt.plot(list(range(len(accuracies))), accuracies[:len(accuracies)], color="r")
        print(f"train acc = {max(accuracies)}")
        plt.xlabel("Minibatch no.")
        plt.ylabel("Attack accuracy")
        plt.show()

        print(f"Done.")

