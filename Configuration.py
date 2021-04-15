import torch
import torch.nn as nn
from Trainer import ModelTrainer

NUM_CLASS_FEATURES = 75
NUM_CLASSES = 10
BATCH_SIZE = 1600
FEATURE_STD_RANGE = (0.5, 1.5)
NUM_EPOCHS = 1000
TRAIN_TEST_RATIO = 9 / 10
SEED = 3

def get_trainer(model: nn.Module):
    # FIXME: should it be weight_decay of 1e-4 or lr_decay?
    # FIXME: use paper's momentum instead of mine
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.01, nesterov=True)
    loss = nn.CrossEntropyLoss()
    return ModelTrainer(model=model, loss_fn=loss, optimizer=optimizer)