import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import glob
import random
from fastai.vision.all import *
from fastai.medical.imaging import *


# load image
def getimage(img_path, mask_path):
        # set file names and path
        folder_name = img_path.split('\\')[-1]
        CT_img_path = os.path.join(img_path, f'{folder_name}__CT.nii.gz')     
        PET_img_path = CT_img_path.replace('__CT.nii.gz', '__PT.nii.gz')
        mask_path = mask_path + '.nii.gz'

        # load images and mask
        CT_image = nib.load(CT_img_path).get_fdata()
        PET_image = nib.load(PET_img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # resize PET image
        pet_r = cv2.resize(PET_image, (CT_image.shape[0], CT_image.shape[1]) )
        # normalize PET image
        PET_image = (pet_r-np.min(pet_r))/(np.max(pet_r)-np.min(pet_r))

        return CT_image, PET_image, mask

def apply_window(img, width=500, center=40):
    # np.interp(a, (a.min(), a.max()), (-1, +1))

    # convert below window to black
    img[img<(center-(width/2))]=center-(width/2)
    # convert above window to white
    img[img>(center+(width/2))]=center+(width/2)

    # normalize image
    img_scaled = np.interp(img, (img.min(), img.max()), (0, +1))
    print(img_scaled.shape)
    print(np.min(img_scaled), np.max(img_scaled))
    return img

def separate_labels(mask):
    gtvp = mask[:]
    gtvn = mask[:]

    # 1 = GTVp, 2 = GTVn
    gtvp = np.where(gtvp <= 1, gtvp, 0)
    gtvn = np.where(gtvn>1, gtvn, 0)

    return gtvp

def save_images(imgs, trgt_path, img_name, number):
    # create directory
    
    # create suffix
    sfx=f"{number:04d}"
    # save CT and PET into images folder
    plt.imsave(f'{trgt_path}/train_images/{img_name}-{sfx}__CT.png', imgs[0], cmap='gray')
    plt.imsave(f'{trgt_path}/train_images/{img_name}-{sfx}__PT.png', imgs[1], cmap='gray')
    plt.imsave(f'{trgt_path}/train_masks/{img_name}-{sfx}.png', imgs[2], cmap='gray')
    # save label into masks folder
    return 

def display_single(image, idx=0, mk=True):
    if mk:
        plt.imshow(image[:,:,idx], cmap='gray')
        plt.show()
        return
    plt.imshow(image, cmap='gray')
    plt.show()


def display_multiple(ct, pet, mask, idx=32):
    n=35
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(15,30))
    ax[0][0].imshow(ct, cmap='gray')
    ax[0][0].set_title('CT Image')
    ax[0][1].imshow(mask, cmap='gray')
    ax[0][1].set_title('GT')
    ax[0][2].imshow(pet, cmap='gray')
    ax[0][2].set_title('PET')
    # # contour
    ax[1][0].imshow(ct, cmap='gray')
    ax[1][0].contour(mask, colors='red')
    ax[1][0].set_title('GT on CT')
    ax[1][1].imshow(pet, cmap='gray')
    ax[1][1].contour(mask, colors='red')
    ax[1][1].set_title('GT on PET')
    ax[1][2].imshow(ct, cmap='gray')
    ax[1][2].contour(pet)
    ax[1][2].contour(mask, colors='red')
    ax[1][2].set_title('All together')
    plt.show()



def test():
    # path to data
    image_dir ='./data/imagesTr'
    mask_dir = './data/labelsTr'
    trgt_dir = './data/jpg' 
    
    image_list = os.listdir(image_dir)
    print(f'found {len(image_list)} images ')
    # print(image_list)

    # get image path
    for img in image_list: 
    # img = image_list[0]
        img_path = os.path.join(image_dir, img)
        # print(img_path)
        
        # get mask path 
        msk_path = os.path.join(mask_dir, img)
        # print(msk_path)
        
        # get image
        ct, pet, mask = getimage(img_path, msk_path)

        for i in range(ct.shape[2]):
            # CT windowing image
            # i=32
            scaled_ct = apply_window(ct[:,:,i], width=500, center=40)
            pet_slice = pet[:,:, i]
            # add function to extract each class from mask
            mask_slice = separate_labels(mask[:,:,i])
            
            
            # add function to save CT, PET, MASK as png
            # save_images((scaled_ct, pet_slice, mask_slice), trgt_dir, img, number=i)
            # display image
            # display_single(scaled_ct, idx=32, mk=False)
            # display_multiple(scaled_ct, pet_slice, mask_slice, idx=i)

if __name__ == '__main__':
    test()