import torch
from model import UNET
# from config import *
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

CHECKPOINT = '230218_B16_160.pth.tar'
DEVICE = 'cpu'

# get the model results keys 
# load model
model = UNET(in_channels=1, out_channels=1).to(DEVICE)
# loss_fn = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() 
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

load_checkpoint(torch.load(f'trained_models/{CHECKPOINT}'), model)

print(model.keys())