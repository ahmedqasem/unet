import torch
# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 160 # 1918 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = './data/ct_pet/train_images'
TRAIN_MASK_DIR = './data/ct_pet/train_labels'
VAL_IMG_DIR = './data/ct_pet/valid_images'
VAL_MASK_DIR = './data/ct_pet/valid_labels'
CHECKPOINT = '230302_B16_160_CTPET.pth.tar'

log_file_name = CHECKPOINT.split('.')[0]
LOG_FILE = f'{log_file_name}'