import torch
import torch.nn as nn
from tqdm import tqdm

NUM_CLASS_FEATURES = 75
NUM_CLASSES = 10
BATCH_SIZE = 400
STD_RANGE = (0.5, 1.5)
SEED = 1

class LinearAttackedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(in_features=NUM_CLASS_FEATURES, out_features=NUM_CLASSES),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layers(x)

class ModelTrainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_batch(self, batch):
        prediction = self.model(batch)
        self.optimizer.zero_grad()
        self.loss_fn(batch, prediction).backward()
        self.optimizer.step()

    def accuracy(self, test_data: torch.Tensor):
        raise NotImplementedError()
        #self.model(test_data)
        #return num_correct / test_data.shape[0]

    def fit(self, train_data: torch.Tensor, test_data: torch.Tensor, num_epochs: int):
        for epoch_num in tqdm(range(num_epochs)):
            print(f"Epoch number {epoch_num}/{num_epochs}:")
            self.train_batch(batch=train_data)
        print(f"Done.  Accuracy: {self.accuracy(test_data)}")

def synthetic_dataset():
    """
    returns random dataset of shape (batch_size, #classes, #features) split into train, test
    """
    feature_std_matrix = torch.diag(torch.distributions.Uniform(*STD_RANGE).sample((NUM_CLASS_FEATURES,)))
    class_means = torch.rand(NUM_CLASSES).unsqueeze(dim=0).unsqueeze(dim=2)
    class_features = torch.matmul(
        feature_std_matrix,
        torch.randn(size=(BATCH_SIZE, NUM_CLASSES, NUM_CLASS_FEATURES)).transpose(1,2)
    ).transpose(1,2) + class_means

    train_data_size = 9 * class_features.shape[0] // 10
    test_data_size = class_features.shape[0] - train_data_size
    return class_features[:train_data_size], class_features[train_data_size:]

def get_trainer(model: nn.Module):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-4, nesterov=True)
    loss = nn.CrossEntropyLoss()
    return ModelTrainer(model=model, loss_fn=loss, optimizer=optimizer)

def main():
    torch.random.manual_seed(SEED)
    train_data, test_data = synthetic_dataset()
    model = LinearAttackedModel()
    trainer = get_trainer(model=model)
    trainer.fit(train_data, test_data)

if __name__ == "__main__":
    main()
