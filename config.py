import torch
# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 2
NUM_WORKERS = 4
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 160 # 1918 originally
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = './data/ct_pet/debug_images'
TRAIN_MASK_DIR = './data/ct_pet/debug_labels'
VAL_IMG_DIR = './data/ct_pet/debug_images'
VAL_MASK_DIR = './data/ct_pet/debug_labels'
CHECKPOINT = 'test_model.pth.tar'