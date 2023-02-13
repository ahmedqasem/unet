import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
from model import UNET
from utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 160 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only_png/train_images'
TRAIN_MASK_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only_png/train_labels'
VAL_IMG_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only_png/valid_images'
VAL_MASK_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only_png/valid_labels'
CHECKPOINT = '220212-ctonlypng.pth.tar'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    # create the augmentation pipeline
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                # mean=[0.0, 0.0, 0.0],
                # std=[1.0, 1.0, 1.0],
                #max_pixel_value=255.0,
                mean=0.5, std=0.5 # if single channel
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                # mean=[0.0, 0.0, 0.0],
                # std=[1.0, 1.0, 1.0],
                # max_pixel_value=255.0,
                mean=0.5, std=0.5 # if single channel
            ),
            ToTensorV2(),
        ],
    )


    # play with this to change to multiclass output
    # change out channels for multi class segmentation
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    # change loss function to cross entropy loss for multi channel
    loss_fn = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # get the loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(f'trained_models/{CHECKPOINT}'), model)

    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f'epoch {epoch+1}/{NUM_EPOCHS}')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimmizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f'trained_models/{CHECKPOINT}')
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader,
        #     model,
        #     folder='./saved_images/', 
        #     device=DEVICE
        # )


if __name__ == '__main__':
    main()
