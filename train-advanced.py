import torch
from torchinfo import summary
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
    save_predictions_as_imgs,
    check_valid_loss
)
from config import *


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    # setup train loss and train accuracy
    train_loss, train_acc = 0, 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        # calculate the loss
        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    # calculate accuracy and dice scores 
    train_acc, train_dc_score = check_accuracy(loader, model, device=DEVICE)
    train_loss = train_loss / len(loader)

    return train_loss, train_acc, train_dc_score


def main():
    # create the augmentation pipeline
    print('building transforms ...')
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                # mean=[0.0, 0.0, 0.0],
                # std=[1.0, 1.0, 1.0],
                # max_pixel_value=255.0,
                mean=0.5, std=0.5 # if single channel
                # dual channel 
                # mean=[0.0, 0.0],
                # std=[1.0, 1.0],
                # max_pixel_value=255.0,
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
                # dual channel 
                # mean=[0.0, 0.0],
                # std=[1.0, 1.0],
                # max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
 

    # play with this to change to multiclass output
    # change out channels for multi class segmentation
    print('creating the model ...')
    model = UNET(in_channels=2, out_channels=1).to(DEVICE)
    # change loss function to cross entropy loss for multi channel
    loss_fn = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # get the loaders
    print('getting loaders...')
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
    
    print('before training')
    
    results = {'train_loss': [],
              'train_acc': [],
              'train_dice': [],
              'valid_loss': [],
              'valid_acc': [],
              'valid_dice': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f'epoch {epoch+1}/{NUM_EPOCHS}')
        train_loss, train_acc, train_dice = train_fn(train_loader, model, optimizer, loss_fn, scaler)

#         # save model
#         checkpoint = {
#             'state_dict': model.state_dict(),
#             'optimmizer': optimizer.state_dict(),
#         }
#         save_checkpoint(checkpoint, filename=f'trained_models/{CHECKPOINT}')

        # check valid loss 
        valid_loss = check_valid_loss(val_loader, model, loss_fn, device=DEVICE)

        # check accuracy 
        valid_acc, valid_dice = check_accuracy(val_loader, model, device=DEVICE)
        print(f' Training Accuracy: {train_acc:.3f} | Validation Accuracy: {valid_acc:.3f}')
        print(f' Training Dice score: {train_dice} | Valid Dice score: {valid_dice}')
        print(f' Training Loss: {train_loss:.4f} | valid loss = {valid_loss:.4f}')
        
        # update results dictionary 
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc.item())
        results['train_dice'].append(train_dice.item())
        results['valid_loss'].append(valid_loss)
        results['valid_acc'].append(valid_acc.item())
        results['valid_dice'].append(valid_dice.item())
    
    # return the filled results
    return results



if __name__ == '__main__':
    results = main()
    print('finished training')
    print(results)
