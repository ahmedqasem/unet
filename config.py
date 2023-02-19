# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 160 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/train_images'
TRAIN_MASK_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/train_labels'
VAL_IMG_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/valid_images'
VAL_MASK_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/valid_labels'
CHECKPOINT = 'trial_2_my_checkpoint.pth.tar'