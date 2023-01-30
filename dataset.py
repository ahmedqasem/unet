import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import torch


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        # change mask values from 255 t0 0-1
        mask[mask==255.0] = 1

        # perform data augmentation using the albemntations library 
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) 
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask



class HecktorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        CT_img_path = os.path.join(self.image_dir, self.images[index])
        # check if it loaded the CT first 
        if CT_img_path[-11:] == '__PT.nii.gz':
           CT_img_path = CT_img_path.replace('__PT.nii.gz', '__CT.nii.gz')
        PET_img_path = os.path.join(self.image_dir, self.images[index].replace('__CT.nii.gz', '__PT.nii.gz'))
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("__PT.nii.gz", ".nii.gz"))
        # load images and mask
        CT_image = nib.load(CT_img_path).get_fdata()
        PET_image = nib.load(PET_img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # resize PET image
        pet_r = cv2.resize(PET_image, (CT_image.shape[0], CT_image.shape[1]) )
        # convert PET CT to 2 channel image
        image = np.moveaxis(np.concatenate((np.expand_dims(CT_image, axis=0), np.expand_dims(pet_r, axis=0)), axis=0), 0, -2)
        
        # change mask values from 255 t0 0-1
        mask[mask==255.0] = 1

        # perform data augmentation using the albemntations library 
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) 
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

def test():
    img = CarvanaDataset(   image_dir='./data/carvana/train_images',
                            mask_dir= './data/carvana/train_masks')
    image, mask = img.__getitem__(1)
    print(f'found image: {image.shape}')
    fig, ax = plt.subplots(ncols=2, nrows=1)
    ax[0].imshow(image)
    ax[1].imshow(mask, cmap='gray')
    plt.show()

def display_image(image, mask, idx=35):
    n=35
    fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(15,30))
    ax[0][0].imshow(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[0][idx], cmap='gray')
    ax[0][0].set_title('CT Image')
    ax[0][1].imshow(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[1][idx], cmap='gray')
    ax[0][1].set_title('PET image')
    ax[1][0].imshow(np.moveaxis(mask, 2, 0)[idx])
    ax[1][0].set_title('Mask Image')
    # contour
    ax[1][1].imshow(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[0][idx], cmap='gray')
    ax[1][1].contour(np.moveaxis(mask, 2, 0)[idx])
    # ax5.imshow(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[1][n], cmap='gray')
    # ax5.contour(np.moveaxis(mask, 2, 0)[n])
    ax[2][0].imshow(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[0][idx], cmap='gray')
    ax[2][0].contour(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[1][idx])
    ax[2][1].imshow(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[1][idx], cmap='gray')
    ax[2][1].contour(np.moveaxis(mask, 2, 0)[idx])
    # all together 
    ax[3][0].imshow(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[0][idx], cmap='gray')
    ax[3][0].contour(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1)[1][idx])
    ax[3][0].contour(np.moveaxis(mask, 2, 0)[idx], colors='red')
    plt.show()

def test_hecktor():
    img = HecktorDataset(   image_dir='./data/train_images',
                            mask_dir= './data/train_masks')
    print(f'found {img.__len__()} images')
    image, mask = img.__getitem__(1)
    print(f'found image: {image.shape}')
    # print(f'moved axis {np.moveaxis(image, 2, 0).shape}')
    # print(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1).shape)
    print(f'found mask: {np.moveaxis(mask, 2, 0)[0].shape}')

    # view sample image and GT
    # display_image(image, mask)


if __name__ == "__main__":
    test_hecktor()
