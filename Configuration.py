import torch
import torch.nn as nn
from Trainer import ModelTrainer, ConvModelTrainer

NUM_CLASS_FEATURES = 75
NUM_CLASSES = 10
BATCH_SIZE = 1600
TRAIN_TEST_RATIO = 9 / 10
TRAIN_DATA_SIZE = int(BATCH_SIZE * TRAIN_TEST_RATIO)
TARGET_PROXY_RATIO = 1 / 2
TARGET_TRAIN_DATA_SIZE = int(TRAIN_DATA_SIZE * TARGET_PROXY_RATIO)
SHADOW_PROXY_RATIO = 1 / 2
SHADOW_TRAIN_DATA_SIZE = int((TRAIN_DATA_SIZE - TRAIN_DATA_SIZE) * SHADOW_PROXY_RATIO)
FEATURE_STD_RANGE = (0.5, 1.5)
NUM_EPOCHS = 1000
SEEDS = [3,4,5,6,7]

def get_linear_trainer(model: nn.Module):
    # FIXME: use paper's momentum instead of mine
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.01, nesterov=True)
    loss = nn.CrossEntropyLoss()
    return ModelTrainer(model=model, loss_fn=loss, optimizer=optimizer)

def get_conv_trainer(model: nn.Module):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.01, nesterov=True)
    loss = nn.L1Loss
    return ConvModelTrainer(model=model, loss_fn=loss, optimizer=optimizer)