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
    save_predictions_as_imgs,
    get_val_loader
)
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 160 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/train_images'
TRAIN_MASK_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/train_masks'
VAL_IMG_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/valid_images_test'
VAL_MASK_DIR = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/jpg/valid_masks'
CHECKPOINT = 'my_checkpoint.pth.tar'


def main():
    # load model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() 
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    load_checkpoint(torch.load(f'trained_models/{CHECKPOINT}'), model)
    
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_loader = get_val_loader(
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    BATCH_SIZE,
    val_transforms,
    NUM_WORKERS,
    PIN_MEMORY
    )

    print(f'len val loader: {len(val_loader)}')
    # print some examples to a folder
    
    # save_predictions_as_imgs(
    #     val_loader,
    #     model,
    #     folder='./saved_images/', 
    #     device=DEVICE
    # )

    # check_accuracy(val_loader, model, device=DEVICE)


    # Iterate over the test dataset and make predictions for each image
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            outputs = model(images)
            print(i, outputs.shape)
            # for j, output in enumerate(outputs):
            #     # save_output(output, f'pred_{i * 32 + j}.png')
            #     check_accuracy(val_loader, model, device=DEVICE)

def make_predictions(model, imagePath, gtPath):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, expand its dimensions, cast it
        # to float data type, and scale its pixel values
        # image = cv2.imread(imagePath, 1)
        image = np.array(Image.open(imagePath).convert('RGB'))
        image = np.expand_dims(image, 0)
        # image = np.expand_dims(image, 0)
        image = np.moveaxis(image, 0, -1)
        image = image.astype("float32") / 255.0 
        print(image.shape)   
 
        
        # find the filename and generate the path to ground truth mask
        # filename = imagePath.split(os.path.sep)[-1]
        # groundTruthPath = os.path.join(Config.Mask_dataset_dir, filename)
        
        # load the ground-truth segmentation mask in grayscale mode and resize it
        gtMask = cv2.imread(gtPath, 0)
        gtMask = cv2.resize(gtMask, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # create a PyTorch tensor, and flash it to the current device
        image = torch.from_numpy(image).to(DEVICE)
        
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image)
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        
        # # filter out the weak predictions and convert them to integers
        # predMask = (predMask > Config.Thresh) * 255
        # predMask = predMask.astype(np.uint8)
        # filename = imagePath.split(os.path.sep)[-1]
        # cv2.imwrite(Config.Base_Out+'\\'+filename, predMask)
        
        # return
        print(gtMask.shape)
        print(predMask.shape)

        return(gtMask, predMask)


if __name__ =='__main__':
    main()