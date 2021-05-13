NUM_CLASS_FEATURES = 20
NUM_CLASSES = 2
BATCH_SIZE = 1000
TRAIN_TEST_RATIO = 9 / 10
TRAIN_DATA_SIZE = int(BATCH_SIZE * TRAIN_TEST_RATIO)
TARGET_PROXY_RATIO = 1 / 2
TARGET_TRAIN_DATA_SIZE = int(TRAIN_DATA_SIZE * TARGET_PROXY_RATIO)
SHADOW_PROXY_RATIO = 1 / 2
ATTACK_TRAIN_DATA_SIZE = TRAIN_DATA_SIZE - TARGET_TRAIN_DATA_SIZE
SHADOW_TRAIN_DATA_SIZE = int(ATTACK_TRAIN_DATA_SIZE * SHADOW_PROXY_RATIO)
FEATURE_STD_RANGE = (0.5, 1.5)
NUM_EPOCHS = 512 # 32
NUM_PATIENCE_EPOCHS = 50
NUM_SHADOW_MODELS = 5*10
MINIBATCH_SIZE = 32
SEEDS = [3,4,5,6,7]

VERBOSE_LINEAR_TRAINING = False
VERBOSE_CONVOLUTION_TRAINING = True

DECAY_RATE = 1 - 10e-4

#DELETE THIS COMMENT
