import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import glob
import random


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
        try:
            assert len(self.images)//2 == len(self.images)/2
            return len(self.images)//2
        except Exception as e:
            print(e)
            return 0
        

    def __getitem__(self, index):
        # set the path to load images from
        img_path = os.path.join(self.image_dir, self.images[index])
        # print(f'loading images from {img_path}')
        slice_name = img_path.split('/')[-1].split('__')[0]
        # print(f'folder name is {slice_name}')

        CT_img_path = os.path.join(self.image_dir, f'{slice_name}__CT.png')
        # print(f'CT image file name is {CT_img_path}')
        
        PET_img_path = CT_img_path.replace('__CT.png', '__PT.png')
        # print(f'loading PET from {PET_img_path}')

        mask_path = os.path.join(self.mask_dir, f'{slice_name}.png')
        # print(f'loading mask from {mask_path}')
        # load images and mask
        CT_image = np.array(Image.open(CT_img_path).convert('L'))
        # print('ct shape is: ', CT_image.shape)
        PET_image = np.array(Image.open(PET_img_path).convert('L'))
        # print('pett shape is: ', PET_image.shape)
        mask = np.array(Image.open(mask_path).convert('L'))
        # print('mask shape is: ', mask.shape)
        
        # normalize
        # print(f'ct values {np.min(CT_image)}, {np.max(CT_image)}')
        # print(f'PET values {np.min(PET_image)}, {np.max(PET_image)}')
        # print(f'mask values {np.min(mask)}, {np.max(mask)}')
        CT_image = CT_image/255
        PET_image = PET_image/255
        mask[mask==255.0] = 1

        # print(f'ct values {np.min(CT_image)}, {np.max(CT_image)}')
        # print(f'PET values {np.min(PET_image)}, {np.max(PET_image)}')
        # print(f'mask values {np.min(mask)}, {np.max(mask)}')


        # convert PET CT to 2 channel image
        image = np.moveaxis(np.concatenate((np.expand_dims(CT_image, axis=0), np.expand_dims(PET_image, axis=0)), axis=0), 0, -1)
        # print('*******', image.shape)

        # perform data augmentation using the albemntations library 
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) 
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

class HecktorDataset_CT(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
        

    def __getitem__(self, index):
        # set the path to load images from
        img_path = os.path.join(self.image_dir, self.images[index])
        # print(f'loading images from {img_path}')
        slice_name = img_path.split('/')[-1].split('__')[0]
        # print(f'folder name is {slice_name}')

        CT_img_path = os.path.join(self.image_dir, f'{slice_name}__CT.png')
        # print(f'CT image file name is {CT_img_path}')
        mask_path = os.path.join(self.mask_dir, f'{slice_name}.png')
        # print(f'loading mask from {mask_path}')
        # load images and mask
        CT_image = np.array(Image.open(CT_img_path).convert('L'), dtype=np.float32)
        # print('ct shape is: ', CT_image.shape)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        # print('mask shape is: ', mask.shape)
        
        # normalize
        CT_image = CT_image/255
        mask[mask==255.0] = 1
        # print(f'ct values {np.min(CT_image)}, {np.max(CT_image)}')
        # print(f'mask values {np.min(mask)}, {np.max(mask)}')
        image=CT_image

        # perform data augmentation using the albemntations library 
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) 
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

def test():
    img = CarvanaDataset(   image_dir='./data/carvana/train_images',
                            mask_dir= './data/carvana/train_masks')
    image, mask = img.__getitem__(0)
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


def display_image2(image, mask, idx=35):

    n=35
    fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(15,30))
    ax[0][0].imshow(image[:,:,0], cmap='gray')
    ax[0][0].set_title('CT Image')
    ax[0][1].imshow(image[:,:,1], cmap='gray')
    ax[0][1].set_title('PET image')
    ax[1][0].imshow(mask, cmap='gray')
    ax[1][0].set_title('Mask Image')
    # # contour
    ax[1][1].imshow(image[:,:,0], cmap='gray')
    ax[1][1].contour(mask)
    ax[2][0].imshow(image[:,:,1], cmap='gray')
    ax[2][0].contour(mask)
    ax[2][1].imshow(image[:,:,0], cmap='gray')
    ax[2][1].contour(image[:,:,1])
    # # all together 
    ax[3][0].imshow(image[:,:,0], cmap='gray')
    ax[3][0].contour(image[:,:,1])
    ax[3][0].contour(mask, colors='red')
    plt.show()

def display_image3(image, mask, idx=35):
    n=35
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,30))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('CT Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask Image')
    # # contour
    ax[2].imshow(image, cmap='gray')
    ax[2].contour(mask)
    # # all together 

    plt.show()

def test_hecktor():

    image_dir ='./data/ct_pet/train_images'
    mask_dir = './data/ct_pet/train_labels'
    img = HecktorDataset(image_dir=image_dir, mask_dir=mask_dir)
    print(f'found {img.__len__()} PET/CT image pairs')
    
    image, mask = img.__getitem__(33)
    print(f'found image: {image.shape}')
    # print(f'moved axis {np.moveaxis(image, 2, 0).shape}')
    # print(np.moveaxis(np.moveaxis(image, 2, 0), 3, 1).shape)
    # print(f'found mask: {np.moveaxis(mask, 2, 0)[0].shape}')

    # view sample image and GT
    # idx = random.randint(0, img.__len__())
    # display_image3(image, mask, idx)


def test_ct():
    image_dir ='E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only_png/train_images'
    mask_dir = 'E:/Datasets/monte_carlo_segmentation/hecktor2022_training/ct_only_png/train_labels'
    img = HecktorDataset_CT(image_dir=image_dir, mask_dir=mask_dir)
    print(f'found {img.__len__()} CT images')
    # load image
    idx = random.randint(0, img.__len__())
    image, mask = img.__getitem__(idx)
    print(f'found image: {image.shape}')
    # show example
    display_image3(image, mask, idx=idx)

if __name__ == "__main__":
    test_hecktor()
