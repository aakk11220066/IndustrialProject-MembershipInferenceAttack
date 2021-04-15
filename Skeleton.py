import torch
from Models import LinearModel
from Configuration import SEED, NUM_EPOCHS, get_trainer
from SyntheticDataset import synthetic_dataset


def main():
    torch.random.manual_seed(SEED)
    train_features, train_labels, test_features, test_labels = synthetic_dataset()
    model = LinearModel()
    trainer = get_trainer(model=model)
    trainer.fit(train_features, train_labels, num_epochs=NUM_EPOCHS)
    print(f"Test accuracy: {trainer.accuracy(test_features, test_labels)}")


if __name__ == "__main__":
    main()
