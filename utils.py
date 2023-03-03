import torch 
import torchvision
from dataset import CarvanaDataset, HecktorDataset_CT, HecktorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir, 
    val_maskdir, 
    batch_size,
    train_transform, 
    val_transform,
    num_workers=4,
    pin_memory=True
):

    train_ds = HecktorDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = HecktorDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def get_val_loader(
    val_dir, 
    val_maskdir, 
    batch_size,
    val_transform,
    num_workers=4,
    pin_memory=True
):

    val_ds = HecktorDataset_CT(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    # print('********************', len(loader)) 
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)#.unsqueeze(1) # unsqueeze because 1 channel images 
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    
    acc = num_correct/num_pixels*100
    dc_score = dice_score/len(loader)
    
    # print(f' Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}')
    # print(f' Dice score: {dice_score/len(loader)}')
    model.train()
    return acc, dc_score


def check_valid_loss(loader, model, loss_fn, device='cuda'):
    valid_loss = 0
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device = device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        # calculate the loss
        valid_loss += loss.item()
    
    valid_loss = valid_loss / len(loader)
    return valid_loss


def save_predictions_as_imgs(
    loader, model, folder='saved_images/', device='cuda'
):
    model.eval() # set the model into eval mode
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        
        with torch.no_grad():
            preds = model(x)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )
        
        # save GT
        yy = y[0].cpu()
        plt.imsave(f"{folder}/gt_{idx}.png", yy, cmap='gray')
        # save image 
        xx = x[0][0].cpu()
        plt.imsave(f"{folder}/image_{idx}.png", xx, cmap='gray')

        model.train()