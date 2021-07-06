# Data descriptions
NUM_CLASS_FEATURES = 20 # Number of features in data
NUM_CLASSES = 2 # Number of classes the target model is to sort the features into
HIDDEN_DIM = 2*NUM_CLASS_FEATURES # Width of hidden dimension of MLP models

# Loss weights
LAYER0_SYNC_LOSS_WEIGHT_PROXY = 1
LAYER2_SYNC_LOSS_WEIGHT_PROXY = 1
LAYER0_SYNC_LOSS_WEIGHT_SHADOW = 1
LAYER2_SYNC_LOSS_WEIGHT_SHADOW = 1

# Training hyperparameters
BATCH_SIZE = 1000
NUM_EPOCHS = 512
NUM_PATIENCE_EPOCHS = 40
NUM_SHADOW_MODELS = 10
MINIBATCH_SIZE = 50
SEEDS = range(2, 10) #

# Ratio of distribution of data between train and test set
TRAIN_TEST_RATIO = 9 / 10
TRAIN_DATA_SIZE = int(BATCH_SIZE * TRAIN_TEST_RATIO)

# Ratio of distribution of train data between target and attacker train set
TARGET_PROXY_RATIO = 1/4 # DONE: Train final proxy model on same number of points as in training
TARGET_TRAIN_DATA_SIZE = int(TRAIN_DATA_SIZE * TARGET_PROXY_RATIO)
ATTACK_TRAIN_DATA_SIZE = TRAIN_DATA_SIZE - TARGET_TRAIN_DATA_SIZE

# Ratio of distributions of data between the shadow and proxy models' training set sizes as well as holdoff data for
# unbiased testing)
SHADOW_PROXY_HOLDOFF_RATIO = (1 / 3, 1 / 3) # DONE: Should split also into non-member points for conv training
SHADOW_TRAIN_DATA_SIZE = int(ATTACK_TRAIN_DATA_SIZE * SHADOW_PROXY_HOLDOFF_RATIO[0])
PROXY_TRAIN_DATA_SIZE = int(ATTACK_TRAIN_DATA_SIZE * SHADOW_PROXY_HOLDOFF_RATIO[1])
HOLDOFF_TRAIN_DATA_SIZE = ATTACK_TRAIN_DATA_SIZE - PROXY_TRAIN_DATA_SIZE - SHADOW_TRAIN_DATA_SIZE

# STD of synthetic database (if in use)
FEATURE_STD_RANGE = (0.5, 1.5)

# Verbosity controls
VERBOSE_REGULAR_TRAINING = False
VERBOSE_CONVOLUTION_TRAINING = True
SHOW_SHADOW_PROXY_LOSS_GRAPHS = False
