NUM_CLASS_FEATURES = 75
NUM_CLASSES = 10
BATCH_SIZE = 1600
TRAIN_TEST_RATIO = 9 / 10
TRAIN_DATA_SIZE = int(BATCH_SIZE * TRAIN_TEST_RATIO)
TARGET_PROXY_RATIO = 1 / 2
TARGET_TRAIN_DATA_SIZE = int(TRAIN_DATA_SIZE * TARGET_PROXY_RATIO)
SHADOW_PROXY_RATIO = 1 / 2
SHADOW_TRAIN_DATA_SIZE = int((TRAIN_DATA_SIZE - TARGET_TRAIN_DATA_SIZE) * SHADOW_PROXY_RATIO)
FEATURE_STD_RANGE = (0.5, 1.5)
NUM_EPOCHS = 1000
NUM_SHADOW_MODELS = 10
SEEDS = [3,4,5,6,7]
