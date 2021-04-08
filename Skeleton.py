import torch
import torch.nn as nn
from Trainer import ModelTrainer
from Models import LinearAttackedModel
from Configuration import SEED, NUM_EPOCHS
from SyntheticDataset import synthetic_dataset


def get_trainer(model: nn.Module):
    # FIXME: should it be weight_decay of 1e-4 or lr_decay?
    # FIXME: use paper's momentum instead of mine
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.01, nesterov=True)
    loss = nn.CrossEntropyLoss()
    return ModelTrainer(model=model, loss_fn=loss, optimizer=optimizer)


def main():
    torch.random.manual_seed(SEED)
    train_features, train_labels, test_features, test_labels = synthetic_dataset()
    model = LinearAttackedModel()
    trainer = get_trainer(model=model)
    trainer.fit(train_features, train_labels, num_epochs=NUM_EPOCHS)
    print(f"Test accuracy: {trainer.accuracy(test_features, test_labels)}")


if __name__ == "__main__":
    main()
