import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
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
IMAGE_WIDTH = 240 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = 'data/carvana/train_images/'
TRAIN_MASK_DIR = 'data/carvana/train_masks/'
VAL_IMG_DIR = 'data/carvana/valid_images/'
VAL_MASK_DIR = 'data/carvana/valid_masks/'
CHECKPOINT = 'my_checkpoint.pth.tar'


# transforms 
train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

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

def test():
    print(f'Dataloaders: {train_loader, val_loader}')
    print(f'length of train_loader: {len(train_loader)} batches of {BATCH_SIZE}')
    print(f'length of val_loader: {len(val_loader)} batches of {BATCH_SIZE}\n')

    # checkout whats inside the training dataloader
    train_features_batch, train_label_batch = next(iter(train_loader))
    print(train_features_batch.shape, train_label_batch.shape)

    # show a sample
    torch.manual_seed(42)
    random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
    img, label = train_features_batch[random_idx], train_label_batch[random_idx]
    
    # plt.imshow(torch.permute(img, (1,2,0)))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20, 10))
    ax1.imshow(torch.permute(img, (1,2,0)))
    ax1.set_title('Image')
    ax2.imshow(label.squeeze(), cmap='gray')
    ax2.set_title('Label')
    plt.show()

if __name__ == '__main__':
    test()